# TP2 - Clustering avec K-Means et K-Medoids
# Module FD1 - Département IA & SD

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("TP N°2 - CLUSTERING DE DONNÉES")
print("=" * 80)

# Fonction pour effectuer le clustering sur un dataset
def analyser_clustering(df, nom_dataset, target_col=None):
    print(f"\n\n{'='*60}")
    print(f"ANALYSE CLUSTERING - {nom_dataset}")
    print(f"{'='*60}")
    
    # 1. PRÉTRAITEMENT DES DONNÉES
    print("\n1. PRÉTRAITEMENT DES DONNÉES:")
    
    # Sélectionner les colonnes numériques
    if target_col and target_col in df.columns:
        X = df.drop(columns=[target_col]).select_dtypes(include=[np.number]).copy()
        y_true = df[target_col] if target_col else None
        print(f"   - Variable cible '{target_col}' exclue pour le clustering")
    else:
        X = df.select_dtypes(include=[np.number]).copy()
        y_true = None
    
    print(f"   - Nombre d'instances: {X.shape[0]}")
    print(f"   - Nombre de features: {X.shape[1]}")
    
    # Gestion des valeurs manquantes
    if X.isnull().sum().sum() > 0:
        print("   - Valeurs manquantes détectées, remplacement par la moyenne...")
        for col in X.columns:
            if X[col].isnull().sum() > 0:
                X[col].fillna(X[col].mean(), inplace=True)
    else:
        print("   - Aucune valeur manquante détectée")
    
    # Normalisation des données
    print("\n2. NORMALISATION:")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    print("   - Normalisation Z-score appliquée")
    print(f"   - Moyennes après normalisation: {X_scaled.mean().round(2).tolist()}")
    print(f"   - Écarts-types après normalisation: {X_scaled.std().round(2).tolist()}")
    
    # 3. DÉTERMINATION DU NOMBRE DE CLUSTERS (COURBE ELBOW)
    print("\n3. DÉTERMINATION DU NOMBRE OPTIMAL DE CLUSTERS:")
    
    inerties = []
    silhouette_scores = []
    K_range = range(2, 11)  # Tester de 2 à 10 clusters
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inerties.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
    
    # Trouver le meilleur k (coude)
    # Calculer les différences d'inertie pour trouver le coude
    diff_inerties = np.diff(inerties)
    diff_diff_inerties = np.diff(diff_inerties)
    
    # Méthode simple : prendre le k où le score de silhouette est maximum
    best_k_silhouette = K_range[np.argmax(silhouette_scores)]
    
    print(f"   - Meilleur k selon silhouette: {best_k_silhouette}")
    print(f"   - Scores de silhouette:")
    for k, score in zip(K_range, silhouette_scores):
        print(f"     k={k}: {score:.4f}")
    
    # Visualisation courbe Elbow
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Courbe d'Elbow
    axes[0].plot(K_range, inerties, 'bo-')
    axes[0].set_xlabel('Nombre de clusters (k)')
    axes[0].set_ylabel('Inertie')
    axes[0].set_title(f'Courbe d\'Elbow - {nom_dataset}')
    axes[0].grid(True)
    # Marquer le meilleur k
    axes[0].axvline(x=best_k_silhouette, color='r', linestyle='--', 
                   label=f'k optimal={best_k_silhouette}')
    axes[0].legend()
    
    # Score de silhouette
    axes[1].plot(K_range, silhouette_scores, 'ro-')
    axes[1].set_xlabel('Nombre de clusters (k)')
    axes[1].set_ylabel('Score de silhouette')
    axes[1].set_title(f'Scores de silhouette - {nom_dataset}')
    axes[1].grid(True)
    axes[1].axvline(x=best_k_silhouette, color='b', linestyle='--',
                   label=f'k optimal={best_k_silhouette}')
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()
    
    # 4. K-MEANS AVEC k OPTIMAL
    print(f"\n4. K-MEANS (k={best_k_silhouette}):")
    
    kmeans = KMeans(n_clusters=best_k_silhouette, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X_scaled)
    
    # Métriques de performance
    kmeans_silhouette = silhouette_score(X_scaled, kmeans_labels)
    kmeans_davies = davies_bouldin_score(X_scaled, kmeans_labels)
    kmeans_inertie = kmeans.inertia_
    
    print(f"   - Inertie intra-cluster: {kmeans_inertie:.2f}")
    print(f"   - Score de silhouette: {kmeans_silhouette:.4f}")
    print(f"   - Indice de Davies-Bouldin: {kmeans_davies:.4f}")
    
    # Distribution des clusters
    unique, counts = np.unique(kmeans_labels, return_counts=True)
    print("   - Distribution des clusters:")
    for cluster, count in zip(unique, counts):
        print(f"     Cluster {cluster}: {count} instances ({count/len(kmeans_labels)*100:.1f}%)")
    
    # 5. K-MEDOIDS
    print(f"\n5. K-MEDOIDS (k={best_k_silhouette}):")
    
    kmedoids = KMedoids(n_clusters=best_k_silhouette, random_state=42, method='pam')
    kmedoids_labels = kmedoids.fit_predict(X_scaled)
    
    # Métriques de performance
    kmedoids_silhouette = silhouette_score(X_scaled, kmedoids_labels)
    kmedoids_davies = davies_bouldin_score(X_scaled, kmedoids_labels)
    kmedoids_inertie = kmedoids.inertia_
    
    print(f"   - Inertie intra-cluster: {kmedoids_inertie:.2f}")
    print(f"   - Score de silhouette: {kmedoids_silhouette:.4f}")
    print(f"   - Indice de Davies-Bouldin: {kmedoids_davies:.4f}")
    
    # Distribution des clusters
    unique, counts = np.unique(kmedoids_labels, return_counts=True)
    print("   - Distribution des clusters:")
    for cluster, count in zip(unique, counts):
        print(f"     Cluster {cluster}: {count} instances ({count/len(kmedoids_labels)*100:.1f}%)")
    
    # 6. COMPARAISON DES PERFORMANCES
    print("\n6. COMPARAISON K-MEANS VS K-MEDOIDS:")
    print(f"   {'Métrique':<20} {'K-Means':<15} {'K-Medoids':<15} {'Meilleur':<10}")
    print(f"   {'-'*60}")
    print(f"   {'Inertie':<20} {kmeans_inertie:<15.2f} {kmedoids_inertie:<15.2f} "
          f"{'K-Means' if kmeans_inertie < kmedoids_inertie else 'K-Medoids'}")
    print(f"   {'Silhouette':<20} {kmeans_silhouette:<15.4f} {kmedoids_silhouette:<15.4f} "
          f"{'K-Means' if kmeans_silhouette > kmedoids_silhouette else 'K-Medoids'}")
    print(f"   {'Davies-Bouldin':<20} {kmeans_davies:<15.4f} {kmedoids_davies:<15.4f} "
          f"{'K-Means' if kmeans_davies < kmedoids_davies else 'K-Medoids'}")
    
    # 7. HISTOGRAMME DES INERTIES
    plt.figure(figsize=(10, 6))
    methods = ['K-Means', 'K-Medoids']
    inerties_compare = [kmeans_inertie, kmedoids_inertie]
    colors = ['skyblue', 'lightcoral']
    
    bars = plt.bar(methods, inerties_compare, color=colors, alpha=0.7)
    plt.ylabel('Inertie')
    plt.title(f'Comparaison des inerties - {nom_dataset}')
    
    # Ajouter les valeurs sur les barres
    for bar, val in zip(bars, inerties_compare):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{val:.2f}', ha='center', va='bottom')
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # 8. VISUALISATION DES CLUSTERS (PCA pour réduction à 2D)
    print("\n7. VISUALISATION DES CLUSTERS (PCA 2D):")
    
    # Réduction dimensionnelle avec PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # K-Means
    scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, 
                               cmap='viridis', alpha=0.6, s=50)
    axes[0].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
                   c='red', marker='X', s=200, edgecolors='black', linewidths=2,
                   label='Centres')
    axes[0].set_title(f'K-Means (k={best_k_silhouette}) - {nom_dataset}')
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    axes[0].legend()
    plt.colorbar(scatter1, ax=axes[0])
    
    # K-Medoids
    scatter2 = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=kmedoids_labels, 
                               cmap='viridis', alpha=0.6, s=50)
    axes[1].scatter(X_pca[kmedoids.medoid_indices_][:, 0], 
                   X_pca[kmedoids.medoid_indices_][:, 1],
                   c='red', marker='X', s=200, edgecolors='black', linewidths=2,
                   label='Médoïdes')
    axes[1].set_title(f'K-Medoids (k={best_k_silhouette}) - {nom_dataset}')
    axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    axes[1].legend()
    plt.colorbar(scatter2, ax=axes[1])
    
    plt.tight_layout()
    plt.show()
    
    # 9. ANALYSE ET CONCLUSION
    print("\n8. ANALYSE DES RÉSULTATS:")
    print("\n   Avantages de K-Means:")
    print("   ✓ Plus rapide et plus efficace pour de grands datasets")
    print("   ✓ Optimise l'inertie intra-cluster")
    if kmeans_silhouette > kmedoids_silhouette:
        print("   ✓ Meilleur score de silhouette (clusters plus compacts)")
    
    print("\n   Avantages de K-Medoids:")
    print("   ✓ Plus robuste aux outliers (utilise des points réels)")
    print("   ✓ Les médoïdes sont interprétables (points existants)")
    if kmedoids_silhouette > kmeans_silhouette:
        print("   ✓ Meilleur score de silhouette (clusters mieux séparés)")
    
    return {
        'best_k': best_k_silhouette,
        'kmeans': {'labels': kmeans_labels, 'silhouette': kmeans_silhouette,
                  'davies': kmeans_davies, 'inertie': kmeans_inertie},
        'kmedoids': {'labels': kmedoids_labels, 'silhouette': kmedoids_silhouette,
                    'davies': kmedoids_davies, 'inertie': kmedoids_inertie}
    }

# Charger les données
print("\nChargement des datasets...")

try:
    diabetes_df = pd.read_csv('/Users/macbok/Desktop/fd_tp/diabetes.csv')
    print("✓ diabetes.csv chargé avec succès")
    
    # Analyse pour diabetes
    results_diabetes = analyser_clustering(diabetes_df, 'DIABETES', target_col='Outcome')
    
except FileNotFoundError:
    print("✗ diabetes.csv non trouvé")

try:
    heart_df = pd.read_csv('/Users/macbok/Desktop/fd_tp/heart.csv')
    print("✓ heart.csv chargé avec succès")
    
    # Analyse pour heart
    results_heart = analyser_clustering(heart_df, 'HEART', target_col='output')
    
except FileNotFoundError:
    print("✗ heart.csv non trouvé")

print("\n" + "="*80)
print("ANALYSE CLUSTERING TERMINÉE")
print("="*80)