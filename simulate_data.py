import pandas as pd
import time
from scipy.stats import qmc
from sklearn.model_selection import ParameterGrid
import numpy as np
import paramiko
from io import StringIO
import os
import warnings
import itertools
from functools import reduce

LH_SAMPLES_SIZE = 150000
LC_SAMPLES_SIZE = 180000
MD_SAMPLES_SIZE = 150000
SEED = 42

class Sampling:

    def __init__(self, minmax_space_constraints=None, mixture_len_constraints=None, total_weight=100, desc_dict=None):
        self.minmax_space_constraints = minmax_space_constraints
        self.mixture_len_constraints = mixture_len_constraints
        self.total_weight = total_weight
        self.desc_dict = desc_dict
        self.rm_desc_dict = {key:val for key, val in self.desc_dict.items() if val['type']=='Material'}
        self.variable_val_desc_dict = {key: val for key, val in self.desc_dict.items() if val['Min'] != val['Max']}
        self.md_rm_desc_dict = {key:val for key, val in self.desc_dict.items() if val['type']=='Material' if val['Min']!=val['Max']}
        self.proc_desc_dict = {key:val for key, val in self.desc_dict.items() if val['type']!='Material'}
        self.fix_vals_dict = {key:val for key, val in self.desc_dict.items() if val['Min'] == val['Max']}
        self.variable_vals_lower_bound = [sub_dict['Min'] for sub_dict in self.variable_val_desc_dict.values()]
        self.variable_vals_upper_bound = [sub_dict['Max'] for sub_dict in self.variable_val_desc_dict.values()]
        self.rm_lower_bound = [sub_dict['Min'] for sub_dict in self.rm_desc_dict.values()]
        self.rm_upper_bound = [sub_dict['Max'] for sub_dict in self.rm_desc_dict.values()]
        self.md_rm_lower_bound = [sub_dict['Min'] for sub_dict in self.md_rm_desc_dict.values()]
        self.md_rm_upper_bound = [sub_dict['Max'] for sub_dict in self.md_rm_desc_dict.values()]
        self.proc_lower_bound = [sub_dict['Min'] for sub_dict in self.proc_desc_dict.values()]
        self.proc_upper_bound = [sub_dict['Max'] for sub_dict in self.proc_desc_dict.values()]
        self.rm_min_max_values = [(low, up) for low, up in zip(self.md_rm_lower_bound, self.md_rm_upper_bound)]
        self.proc_min_max_values = [(low, up) for low, up in zip(self.proc_lower_bound, self.proc_upper_bound)]
        self.column_names = list(self.variable_val_desc_dict.keys())
        self.rm_cols = list(self.rm_desc_dict.keys())
        self.md_rm_cols = list(self.md_rm_desc_dict.keys())
        self.proc_cols = list(self.proc_desc_dict.keys())
        self.fix_val_cols = list(self.fix_vals_dict.keys())
        self.fix_values = [value['Min'] for value in self.fix_vals_dict.values()]#list(self.fix_vals_dict.keys())
        self.combinations_levels = self.get_combinations(length=len(self.column_names), minmax_values=self.variable_val_desc_dict.values())

    @staticmethod
    def get_combinations(length, minmax_values):
        non_fix_cols_len = length
        p_level_value = 33 if non_fix_cols_len <= 3 else 11 if non_fix_cols_len in [4, 5] else 5
        levels = [list(np.linspace(sub_dict['Min'], sub_dict['Max'], p_level_value)) for sub_dict in
                  minmax_values if sub_dict['Min']!=sub_dict['Max']]
        return levels

    @staticmethod
    def check_timeout(start_time):
        timeout = 60
        if time.time() - start_time > timeout:
            raise TimeoutError("Sampling exceeded the time limit.")

    @staticmethod
    def apply_binary_space_constraints(df, constraints):
        if not constraints:
            return df
        mask = pd.Series([True] * len(df), index=df.index)
        for constraint in constraints:
            mask &= df.astype(bool).astype(int).query(constraint).any(axis=1)
        df = df[mask]
        return df

    @staticmethod
    def apply_minmax_space_constraints(df, constraints):
        if not constraints:
            return df
        for constraint in constraints:
            if df.empty:
                return df
            df = df.query(constraint)
        return df

    def latin_hypercube_samples(self, sample_size, constraints):
        param_ranges = [(self.variable_vals_lower_bound[i], self.variable_vals_upper_bound[i]) for i in range(len(self.variable_vals_lower_bound))]
        sampler = qmc.LatinHypercube(d=len(param_ranges), seed=SEED)
        lh_samples = sampler.random(n=sample_size)
        lh_samples = pd.DataFrame(lh_samples, columns=self.column_names)
        for i in range(len(self.variable_vals_lower_bound)):
            lh_samples.iloc[:, i] = self.variable_vals_lower_bound[i] + lh_samples.iloc[:, i] * (self.variable_vals_upper_bound[i] - self.variable_vals_lower_bound[i])
        lh_samples = self.apply_minmax_space_constraints(df = lh_samples, constraints=constraints)
        return lh_samples.round(2)

    def regular_grid_samples(self, sample_size, constraints):
        bounds = {self.column_names[i]: (self.variable_vals_lower_bound[i], self.variable_vals_upper_bound[i]) for i in range(len(self.variable_vals_lower_bound))}
        num_pts_per_dim = max(1, np.floor(sample_size ** (1 / len(bounds))).astype(int))
        start_time = time.time()
        while True:
            self.check_timeout(start_time)
            param_grid = {name: np.linspace(bnd[0], bnd[1], num=num_pts_per_dim) for name, bnd in bounds.items()}
            grid_samples = list(ParameterGrid(param_grid))
            rg_samples_df = pd.DataFrame(grid_samples)[self.column_names]
            rg_samples_df = self.apply_minmax_space_constraints(df = rg_samples_df, constraints=constraints)
            if len(rg_samples_df) >= sample_size:
                rg_samples_df = rg_samples_df.iloc[:sample_size]
                break
            elif len(rg_samples_df) == 0:
                num_pts_per_dim = max(1, num_pts_per_dim // 2)
                if num_pts_per_dim == 1:
                    raise ValueError("No samples met the constraints. Consider adjusting the constraints or bounds.")
            else:
                break
        return rg_samples_df.round(2)

    @staticmethod
    def get_mixture_design(m):
        hostname = st.secrets['hostname']
        username = st.secrets['username']
        password = st.secrets['password']
        transport = paramiko.Transport(hostname)
        transport.connect(username=username, password=password)
        sftp = paramiko.SFTPClient.from_transport(transport)
        file_name = f'm_{m}_p_'
        remote_dir_path = f'/home/workdir_team/mixture_design/md_lookup/'
        files = sftp.listdir_attr(remote_dir_path)
        files_found = [f for f in files if f.filename.startswith(file_name)]
        df = pd.DataFrame()
        if files_found:
            latest_file = max(files_found, key=lambda f: f.st_mtime)
            remote_csv_path = os.path.join(remote_dir_path, latest_file.filename)
            try :
                sftp.stat(remote_csv_path)
                with sftp.open(remote_csv_path, 'r') as remote_file:
                    csv_content = remote_file.read().decode()
                sftp.close()
                transport.close()
            except FileNotFoundError :
                raise f"File '{file_name}' not found on server"
            csv_data = StringIO(csv_content)
            df = pd.read_csv(csv_data)
        else :
            pass
        return df

    @staticmethod
    def convert_to_best_numeric_type(a):
        try:
            num = float(a)
            if num.is_integer():
                return int(num)
            return num
        except ValueError:
            return a

    def mixture_design_samples(self, constraints, sample_size):
        md_samples = self.get_mixture_design(len(self.md_rm_cols))
        if md_samples is None:
            md_samples = pd.DataFrame(columns=self.md_rm_cols)
            warnings.warn("'no mixture_design samples found'")
        else :
            md_samples.columns = self.md_rm_cols
        try :
            md_samples = self.apply_binary_space_constraints(df=md_samples, constraints=constraints)
        except Exception as e:
            raise f"Erreur lors de l'application de la contrainte : {e}"
        if any('==' in x for x in constraints):
            equal_constraints = list(filter(lambda x: '==' in x, constraints))
            for constraint in equal_constraints:
                constraint_cols = [col for col in md_samples.columns if col in constraint]
                constraint_number = self.convert_to_best_numeric_type(constraint.split('==')[-1])
                md_samples_sous_plan1_normalized = self.get_mixture_design(len(constraint_cols))
                if md_samples_sous_plan1_normalized is None:
                    pass
                else:
                    md_samples_sous_plan1_normalized.columns = constraint_cols
                    md_samples_sous_plan1_denormalized = md_samples_sous_plan1_normalized * (constraint_number / 100)

                    # s'il y a des variables qui ne sont pas des matières premières, on leur crée des samples avec lh
                    if len(self.rm_min_max_values) > len(constraint_cols):
                        # chercher les limites uniquement pour les éléments qui ne sont pas des matières premières
                        lh_param_ranges = self.rm_min_max_values[len(constraint_cols):]
                        sampler = qmc.LatinHypercube(d=len(lh_param_ranges), seed=SEED)

                        # Pour garantir la reproductibilité des samples
                        all_samples = sampler.random(MD_SAMPLES_SIZE)
                        np.random.seed(SEED)
                        indices = np.random.choice(MD_SAMPLES_SIZE, size=50, replace=False)

                        md_samples_sous_plan2_samples = all_samples[indices]
                        md_samples_sous_plan2_samples_scaled = qmc.scale(md_samples_sous_plan2_samples,
                                                                         [elem[0] for elem in lh_param_ranges],
                                                                         [elem[-1] for elem in lh_param_ranges])
                        md_samples_sous_plan2 = pd.DataFrame(md_samples_sous_plan2_samples_scaled, columns=list(filter(
                            lambda x: x not in constraint_cols, self.md_rm_cols)))

                        md_samples = pd.merge(md_samples_sous_plan1_denormalized, md_samples_sous_plan2,
                                                 how='cross')
                    else:
                        md_samples = md_samples_sous_plan1_denormalized.copy()

        if len(self.fix_values) == 0 :
            mixture_design_bloc_weight = self.total_weight
        else :
            mixture_design_bloc_weight = self.total_weight - sum(self.fix_values)
        md_samples = md_samples * mixture_design_bloc_weight

        return md_samples.iloc[:sample_size]

    @staticmethod
    def generate_combinations(levels, total, nb_row, include_flags):
        """
        Generate combinations of given levels for n-1 variables
        with specified inclusion flags, such that the sum of selected variables
        plus the nth variable equals a specified total, with a timeout feature.
        :param levels: A list of lists, where each inner list contains the levels for each variable (except the last).
        :param total: The total sum that each combination should equal.
        :param nb_row: The number of combinations to generate.
        :param include_flags: A list of booleans indicating whether to include each variable in the total.
        :return: A pandas DataFrame containing valid combinations (up to nb_row) or max combinations found within the timeout.
        """

        valid_combinations = []
        # Use itertools.product to generate combinations for the first n-1 variables
        for combination in itertools.product(*levels):
            # Calculate the sum only for included variables
            selected_sum = sum(comb for comb, include in zip(combination, include_flags) if include)
            last_value = total - selected_sum

            if last_value >= 0:  # Ensure the last variable is non-negative
                valid_combinations.append(combination + (last_value,))
                if len(valid_combinations) >= nb_row:  # Stop if we reach nb_row
                    break
        df = pd.DataFrame(valid_combinations, columns=[f'C_{i}' for i in range(len(levels) + 1)])
        df = df.drop_duplicates()
        return df

    def level_combinations_samples(self, levels, total_weight, sample_size):
        nb_factors = len(self.column_names)
        include_flags = [True] * (nb_factors-1)
        lc_samples = self.generate_combinations(levels=levels, total=total_weight, nb_row=sample_size,
                                                include_flags=include_flags)
        lc_samples.drop(f"C_{nb_factors}", axis=1, inplace=True)
        lc_samples.columns = self.column_names
        return lc_samples

    def collapse_data(self):
        lh_samples = self.latin_hypercube_samples(sample_size=LH_SAMPLES_SIZE, constraints=self.minmax_space_constraints)
        lc_samples = self.level_combinations_samples(sample_size=LC_SAMPLES_SIZE, total_weight=self.total_weight, levels=self.combinations_levels)
        md_samples = self.mixture_design_samples(sample_size=MD_SAMPLES_SIZE, constraints=self.mixture_len_constraints)
        for col in lh_samples.columns:
            if col not in md_samples.columns:
                md_samples[col] = lh_samples[col]
        concat_samples = pd.concat([lh_samples, lc_samples, md_samples]).round(6).reset_index(drop=True)
        return concat_samples

    def filter_by_constraints(self, samples):
        constraints_filtered_samples = samples.copy()
        if self.minmax_space_constraints:
            temp_df = constraints_filtered_samples.copy()
            for col in self.fix_val_cols:
                temp_df[col] = self.fix_values[self.fix_val_cols.index(col)]
            temp_df = self.apply_minmax_space_constraints(temp_df, constraints=self.minmax_space_constraints)
            constraints_filtered_samples = constraints_filtered_samples.iloc[temp_df.index].reset_index(drop=True)
        if self.mixture_len_constraints:
            temp_df = constraints_filtered_samples.copy()
            for col in self.fix_val_cols:
                temp_df[col] = self.fix_values[self.fix_val_cols.index(col)]
            temp_df = self.apply_binary_space_constraints(temp_df, constraints=self.mixture_len_constraints)
            constraints_filtered_samples = constraints_filtered_samples.iloc[temp_df.index].reset_index(drop=True)
        return constraints_filtered_samples

    def filter_by_min_max_values(self, samples):
        minmax_filtered_samples = samples.copy()
        checker = []
        for col in self.md_rm_cols:
            sub_checker_rm = [minmax_filtered_samples[col] >= self.rm_min_max_values[self.md_rm_cols.index(col)][0],
                              minmax_filtered_samples[col] <= self.rm_min_max_values[self.md_rm_cols.index(col)][-1]]
            result_rm = reduce(lambda x, y: x & y, sub_checker_rm)
            checker.append(result_rm)

        for col in self.proc_cols:
            sub_checker_proc = [minmax_filtered_samples[col] >= self.proc_min_max_values[self.proc_cols.index(col)][0],
                               minmax_filtered_samples[col] <= self.proc_min_max_values[self.proc_cols.index(col)][-1]]
            result_proc = reduce(lambda x, y: x & y, sub_checker_proc)
            checker.append(result_proc)
        #if self.nb_levels_list :
        #    all_cols = self.md_rm_cols + self.proc_cols
        #    for col in all_cols:
        #        sub_checker_lev = [minmax_filtered_samples[col].isin(self.combinations_levels[all_cols.index(col)])]
        #        result_lev = reduce(lambda x, y: x & y, sub_checker_lev)
        #        checker.append(result_lev)
        minmax_filtered_samples = minmax_filtered_samples[reduce(lambda x, y: x & y, checker)]
        return minmax_filtered_samples

    def get_samples(self):
        all_samples_df = self.collapse_data()
        filtered_samples_by_constraints = self.filter_by_constraints(samples = all_samples_df)
        filtered_samples_by_minmax = self.filter_by_min_max_values(samples = filtered_samples_by_constraints)
        final_samples_df = filtered_samples_by_minmax.copy()
        for col in self.fix_val_cols :
            col_position = list(self.desc_dict.keys()).index(col)
            final_samples_df.insert(col_position, col, self.fix_values[self.fix_val_cols.index(col)])
        return final_samples_df
