import umap
from sklearn.cluster import KMeans

def plot_project(x, save=None):

    x1 = x[:,0]
    x2 = x[:,1]

    fig=plt.figure(figsize=(6,6))
    plt.scatter(x1, x2, marker='.', c='black', alpha=0.4)


    if save!=None:
        plt.savefig(save)
    plt.show()
    
def plot_cluster(x, labels, save=None):
    x1 = x[:,0]
    x2 = x[:,1]

    fig=plt.figure(figsize=(8,6))

    c = plt.scatter(x1, x2, c=labels, cmap='gist_rainbow', marker='.')
    plt.colorbar(boundaries=np.arange(max(labels)+2)-0.5).set_ticks(np.arange(max(labels)+1))

    plt.scatter(x1, x2, c=labels, cmap='gist_rainbow', marker='.')
    if save!=None:
        plt.savefig(save)


    plt.show()

def umap_embedding(z, nei=50, save=None):
    fit = umap.UMAP(
        n_neighbors=nei,
        min_dist=0.05,
        n_components=2,
        metric='euclidean'
        )
    umaped = fit.fit_transform(z)
    if save!=None:
        np.save('%s/z_umap'%self.path, umaped)

    return umaped

def get_kmeans_labels(x, n_clusters):
    kmeans_labels = KMeans(init="k-means++", n_clusters=n_clusters, random_state=0, max_iter = 5).fit(x).labels_
    return kmeans_labels



def plot_g_info(lid):
    g_img = imgset[:5000][l==lid]
    g_val = valset[:5000][l==lid]
    g_label = label[:5000][l==lid]
    plt.figure(figsize=(6,2))
    plt.title(lid)
    plt.hist(g_label[:,2], bins=50)
    plt.show()
    for i in range(5):
        plt.subplot(1,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(g_img[i][:,:,0])

    plt.show()

    plt.figure(figsize=(10,1))
    for i in range(5):
        plt.subplot(1,5,i+1)
        plt.plot(g_val[i][:,0])
    plt.show()
