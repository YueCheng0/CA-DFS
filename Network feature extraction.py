import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors
import warnings
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
warnings.filterwarnings('ignore')



class OptimizedSampleSpecificNetwork:
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.consensus_network = None
        self.sample_similarity_matrix = None
        self.residual_matrix = None
        self.X_network = None
        
    def load_cdd_network(self, cdd_file, threshold=0.7, max_edges=5000):

        cdd_df = pd.read_csv(cdd_file, index_col=0)  
        self.original_gene_names = cdd_df.columns.tolist() 
        self.gene_names = self.original_gene_names.copy()  
        n_genes = len(self.gene_names)
        cdd_matrix = cdd_df.values
        edges = []
        for i in tqdm(range(n_genes), desc="提取CDD边"):
            for j in range(n_genes):
                if i != j and abs(cdd_matrix[i, j]) > threshold:
                    edges.append((i, j, cdd_matrix[i, j]))
        
        edges_sorted = sorted(edges, key=lambda x: abs(x[2]), reverse=True)
        selected_edges = edges_sorted[:max_edges]
        self.consensus_network = [(i, j) for i, j, w in selected_edges]
        return self.consensus_network
    
    def load_sample_similarity(self, similarity_file):

        similarity_df = pd.read_csv(similarity_file, index_col=0)     
        self.sample_names = similarity_df.index.tolist()
        self.sample_similarity_matrix = similarity_df.values
        
        self.sample_similarity_matrix = (self.sample_similarity_matrix + 
                                       self.sample_similarity_matrix.T) / 2
        
        min_val = np.min(self.sample_similarity_matrix)
        max_val = np.max(self.sample_similarity_matrix)

        if min_val < 0 or max_val > 1:
            self.sample_similarity_matrix = (self.sample_similarity_matrix - min_val) / (max_val - min_val)      
        return self.sample_similarity_matrix
    
    def load_expression_data(self, expression_file):
        expression_df = pd.read_csv(expression_file, index_col=0)        
        expr_sample_names = expression_df.index.tolist()
        expr_gene_names = expression_df.columns.tolist()
        
        if hasattr(self, 'sample_names'):
            common_samples = list(set(self.sample_names) & set(expr_sample_names))
           
            if len(common_samples) == 0:
                raise ValueError("error")
                
            if len(common_samples) < len(self.sample_names):
                self.sample_names = common_samples
                sample_indices = [expr_sample_names.index(s) for s in common_samples]
                expression_df = expression_df.iloc[sample_indices]
        
        if hasattr(self, 'gene_names'):         
            common_genes = list(set(self.gene_names) & set(expr_gene_names))
            if len(common_genes) == 0:
                cdd_genes_lower = [g.lower() for g in self.gene_names]
                expr_genes_lower = [g.lower() for g in expr_gene_names]
                common_genes_lower = list(set(cdd_genes_lower) & set(expr_genes_lower))
                
                if len(common_genes_lower) > 0:
                    gene_mapping = {}
                    for expr_gene in expr_gene_names:
                        if expr_gene.lower() in common_genes_lower:
                            cdd_gene = self.gene_names[cdd_genes_lower.index(expr_gene.lower())]
                            gene_mapping[cdd_gene] = expr_gene
                    
                    common_genes = list(gene_mapping.keys())
                    expression_df = expression_df.rename(columns={v: k for k, v in gene_mapping.items()})
                    expression_df = expression_df[common_genes]
                else:
                    raise ValueError("error")
            
            if len(common_genes) < len(self.gene_names):
                self.gene_names = common_genes
                expression_df = expression_df[common_genes]
                gene_to_index = {gene: idx for idx, gene in enumerate(self.gene_names)}
                filtered_network = []
                
                for tf_idx, target_idx in self.consensus_network:
                    if tf_idx < len(self.original_gene_names) and target_idx < len(self.original_gene_names):
                        tf_gene = self.original_gene_names[tf_idx]
                        target_gene = self.original_gene_names[target_idx]
                        
                        if tf_gene in self.gene_names and target_gene in self.gene_names:
                            new_tf_idx = self.gene_names.index(tf_gene)
                            new_target_idx = self.gene_names.index(target_gene)
                            filtered_network.append((new_tf_idx, new_target_idx))
                
                self.consensus_network = filtered_network

        
        self.expression_data = expression_df.value
        
        if len(self.consensus_network) == 0:
            raise ValueError("error")
            
        return self.expression_data
    
    def compute_sample_specific_residuals(self, k_neighbors=15, max_samples=None):
        if self.consensus_network is None or len(self.consensus_network) == 0:
            raise ValueError("error")
        if self.sample_similarity_matrix is None:
            raise ValueError("error")
        if not hasattr(self, 'expression_data'):
            raise ValueError("error")
        
        n_samples = self.expression_data.shape[0]
        n_edges = len(self.consensus_network)
        
        if max_samples is not None:
            n_samples = min(n_samples, max_samples)
        
        if self.expression_data.shape[1] == 0:
            raise ValueError("error")
        
        scaler = StandardScaler()
        X_expr_scaled = scaler.fit_transform(self.expression_data)
        
        distance_matrix = 1 - self.sample_similarity_matrix

        if np.min(distance_matrix) < 0:
            print(f"{np.min(distance_matrix):.6f}")
            distance_matrix[distance_matrix < 0] = 0

        if not np.allclose(distance_matrix, distance_matrix.T):
            distance_matrix = (distance_matrix + distance_matrix.T) / 2
        

        self.residual_matrix = np.zeros((n_samples, n_edges))
        
        try:
            nn = NearestNeighbors(n_neighbors=k_neighbors, metric='precomputed')
            nn.fit(distance_matrix)
        except Exception as e:
            nn = NearestNeighbors(n_neighbors=k_neighbors, metric='euclidean')
            nn.fit(self.expression_data)
        
        
        # 使用进度条处理每条边
        for edge_idx in tqdm(range(n_edges), desc="处理调控边"):
            tf_idx, target_idx = self.consensus_network[edge_idx]
            
            for sample_i in range(n_samples):
                try:
                    if hasattr(nn, '_fit_X') and nn._fit_X is not None:
                        distances, neighbor_indices = nn.kneighbors(
                            nn._fit_X[sample_i].reshape(1, -1) if nn._fit_X.shape[1] == self.expression_data.shape[1] 
                            else distance_matrix[sample_i].reshape(1, -1)
                        )
                    else:
                        from sklearn.metrics.pairwise import pairwise_distances
                        if hasattr(self, 'expression_data'):
                            distances = pairwise_distances(
                                self.expression_data[sample_i].reshape(1, -1), 
                                self.expression_data
                            )[0]
                        else:
                            distances = distance_matrix[sample_i]
                        
                        neighbor_indices = np.argsort(distances)[:k_neighbors+1] 
                        distances = distances[neighbor_indices]
                    
                    neighbor_indices = neighbor_indices[0] if len(neighbor_indices.shape) > 1 else neighbor_indices
                    neighbor_indices = neighbor_indices[neighbor_indices != sample_i]
                    
                    if len(neighbor_indices) < 3:
                        neighbor_indices = np.delete(np.arange(len(self.sample_names)), sample_i)
                        if len(neighbor_indices) > k_neighbors:
                            neighbor_indices = neighbor_indices[:k_neighbors]
                    X_train = X_expr_scaled[neighbor_indices, tf_idx].reshape(-1, 1)
                    y_train = X_expr_scaled[neighbor_indices, target_idx]
                    
                    model = Ridge(alpha=1.0, random_state=self.random_state)
                    model.fit(X_train, y_train)
                    
                    X_test = [[X_expr_scaled[sample_i, tf_idx]]]
                    y_pred = model.predict(X_test)[0]
                    y_actual = X_expr_scaled[sample_i, target_idx]
                    
                    residual = abs(y_actual - y_pred)
                    self.residual_matrix[sample_i, edge_idx] = residual
                    
                except Exception as e:

                    self.residual_matrix[sample_i, edge_idx] = 0
        
        return self.residual_matrix
    
    def build_sample_specific_networks(self):
        if self.residual_matrix is None:
            raise ValueError("error")
        
        n_samples = self.residual_matrix.shape[0]
        n_genes = len(self.gene_names)
        
        self.sample_networks = []
        
        for sample_i in tqdm(range(n_samples), desc="构建样本网络"):
            adj_matrix = np.zeros((n_genes, n_genes))
            
            for edge_idx, (tf_idx, target_idx) in enumerate(self.consensus_network):
                residual = self.residual_matrix[sample_i, edge_idx]
                adj_matrix[tf_idx, target_idx] = residual
            
            self.sample_networks.append(adj_matrix)
        
        return self.sample_networks
    
    def extract_network_features(self):
        if not hasattr(self, 'sample_networks'):
            raise ValueError("error")
        
        n_samples = len(self.sample_networks)
        n_genes = len(self.gene_names)
        
        self.X_network = np.zeros((n_samples, 2 * n_genes))
        
        for sample_i, adj_matrix in enumerate(self.sample_networks):
            out_degree = np.sum(adj_matrix, axis=1)
            in_degree = np.sum(adj_matrix, axis=0)
            
            self.X_network[sample_i, :n_genes] = out_degree
            self.X_network[sample_i, n_genes:] = in_degree
        
        scaler = StandardScaler()
        self.X_network = scaler.fit_transform(self.X_network)
        
        self.network_feature_names = (
            [f'RegStrength_{gene}' for gene in self.gene_names] +
            [f'TargetSens_{gene}' for gene in self.gene_names]
        )
        
        return self.X_network
    
    def save_network_features(self, output_file):
        if self.X_network is None:
            raise ValueError("error")
        
        feature_df = pd.DataFrame(
            self.X_network,
            index=self.sample_names[:self.X_network.shape[0]],
            columns=self.network_feature_names
        )
        
        feature_df.to_csv(output_file)
        return feature_df
    
    def run_optimized_pipeline(self, cdd_file, similarity_file, expression_file,
                             cdd_threshold=0.7, k_neighbors=15, max_edges=2000, 
                             max_samples=None, output_file="optimized_network_features.csv"):

        try:
            self.load_cdd_network(cdd_file, cdd_threshold, max_edges)
            self.load_sample_similarity(similarity_file)
            self.load_expression_data(expression_file)
            self.compute_sample_specific_residuals(k_neighbors, max_samples)
            self.build_sample_specific_networks()
            self.extract_network_features()
            self.save_network_features(output_file)            
            return self.X_network
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return None

def main():
    
    analyzer = OptimizedSampleSpecificNetwork(random_state=42)
    
    # 运行优化版流程
    X_network_features = analyzer.run_optimized_pipeline(
        cdd_file="network/f1_0_4_renamed.csv",  
        similarity_file="Sample_similarity_network_matrix.csv",
        expression_file="omics_0_feature_select_matrix.csv",
        cdd_threshold=0.7,
        k_neighbors=20,
        max_edges=20000,
        max_samples=None,
        output_file="optimized_network_features.csv"
    )
    df_cleaned = X_network_features.loc[:, ~(X_network_features == 0).all()]
    scaler_minmax = MinMaxScaler()
    normalized_array = scaler_minmax.fit_transform(df_cleaned)
    df_normalized = pd.DataFrame(
    normalized_array,
    columns=df_cleaned.columns,
    index=df_cleaned.index)
    df_normalized.to_csv('optimized_network_features_normalized.csv')

if __name__ == "__main__":
    main()