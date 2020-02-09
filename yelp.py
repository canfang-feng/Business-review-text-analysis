#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud,ImageColorGenerator
from sklearn.preprocessing import StandardScaler


# In[3]:


#### import the data sets of rating and review text
data=pd.read_csv("1000_random_reviews.csv")
text=pd.read_csv("bow1000.csv")


# In[ ]:


### Exploratory analysis


# In[5]:


# use histogram to visual the distribution of the star ratings.
bin_edges=[0.5,1.5,2.5,3.5,4.5,5.5]
sns.set()

fig=plt.hist(data["stars"],bins=bin_edges)
plt.xlabel("star rating")
plt.ylabel("numbers of the reviews")
plt.show()
plt.savefig("img/star_rating")


# In[6]:


#  Use the column “stars” to select the reviews with star rating equal or higher than 4,
#  and generate a “wordcloud" plot for 100 most frequent words of the selected reviews. 

h_text=text.loc[data["stars"]>=4]
h_word=h_text.sum(axis=0,skipna=True).sort_values(ascending=False)[:100]
print(h_word.head(10))
wordcloud=WordCloud(background_color="white").generate(str(list(h_word.index)))
plt.figure()
plt.title("wordcloud of high rating reviews")
plt.imshow(wordcloud,interpolation="bilinear")
plt.axis("off")
plt.show()
wordcloud.to_file("img/high_review.png")


# In[7]:


# select the reviews with star rating lower or equal to 2 and 
# generate a “wordcloud" plot for 100 most frequent words of the selected reviews. 
l_text=text.loc[data["stars"]<=2]
l_word=l_text.sum(axis=0,skipna=True).sort_values(ascending=False)[:100]
print(l_word.head(10))
wordcloud2=WordCloud(background_color="white").generate(str(list(l_word.index)))
plt.figure()
plt.imshow(wordcloud2,interpolation="bilinear")
plt.axis("off")
plt.title("wordcloud of low rating reviews")
plt.show()
wordcloud2.to_file("img/low_review.png")


# In[8]:


# correlation between features and feature-labels
corr=data.corr()
print(corr)
ax=sns.heatmap(corr,vmin=-1,vmax=1,center=0,cmap=sns.diverging_palette(20, 220, n=200),square=True)
ax.set_xticklabels(ax.get_xticklabels(),rotation=45,horizontalalignment='right')
ax.set_title("The correlation between cool,funny,useful and stars")


# In[10]:


# write a function of PCA
def pca_df(self, eigen_info=False, variance_info=False):
    """
    Builds the dataframe with the new coordinates after PCA.
    :param self: Food object.
    :param eigen_info: bool, when True prints the 3 largest eigenvalues and their associated eigenvectors.
            Default: False
    :param variance_info: bool, when True prints information on variance coverage in 3 dimensions after PCA. 
            It also shows that PCA preserves the total variance, by showing the plot of the cumulative
            explained variance and the cumulative variance of original components.
            Default: False
    :return: the dataframe
    """

    # Take only columns with the numerical values from the standardized data frame
    new=self.drop(["business_id","date","review_id","text","user_id"],axis=1)
    X_std=StandardScaler().fit_transform(new)
    mean_vec=np.mean(X_std,axis=0)
    #cov_mat=(X_std-mean_vec).T.dot((X_std-mean_vec))/(X_std.shape[0]-1)
    cov_mat=np.cov(X_std.T)
    eig_vals,eig_vecs=np.linalg.eig(cov_mat)
    # Create new empty data frame for the PCA
    pca_df = pd.DataFrame()
    
    # Add columns with food names and labels
    pca_df['name'] = self['review_id']

   

    # Build dataframe with new coordinates after PCA
    # TO COMPLETE: write the matrix multiplication (@) between the ith eigenvector and 
    # the transposed standardized columns. 
    # The ith eigenvector corresponds to the ith column of the eigenvector matrix.
    for i in range(eig_vecs.shape[1]):
    
        pca_df['PCA_cmp' + str(i + 1)] =eig_vecs[:,i]@(X_std.T)

    # Print info about eigenValues and eigenVectors
    if eigen_info:
        
        print('--> Here below the 3 largest eigenvalues and their eigenvectors:')
        print('lambda_1 (largest eigenvalue)               =', eig_vals[0])
        print('u_1 (eigenvector corresponding to lambda_1) =', eig_vecs[:, 0])
        print('lambda_2 (2nd largest eigenvalue)           =', eig_vals[1])
        print('u_2 (eigenvector corresponding to lambda_2) =', eig_vecs[:, 1])
        print('lambda_3 (3rd largest eigenvalue)           =', eig_vals[2])
        print('u_3 (eigenvector corresponding to lambda_3) =', eig_vecs[:, 2])

    
    return pca_df


# In[11]:


pca_df(data, eigen_info=True, variance_info=True).sample(5)


# In[14]:


new=data.drop(["business_id","date","review_id","text","user_id"],axis=1)
Z=StandardScaler().fit_transform(text)


# In[15]:


# Use PCA for BoW1000.csv to plot the data as a two-dimensional scatter plot. 
# How many PCA components are need to reconstruct 50% of the original data?

from sklearn.decomposition import PCA
pca = PCA().fit(Z)
W_pca = pca.components_
X = np.dot(Z,W_pca.T)

com=np.cumsum(pca.explained_variance_ratio_)
print()
fig1=plt.plot(com)
plt.xlabel("number of components")
plt.ylabel("cumulative explained variance")
ax=plt.gca()
ax.set_ylim([0,0.5])
ax.set_xlim([0,107])
plt.show()

plt.savefig("img/cumulative explained variance")


X_PC12 = X[:,[0,1]]

plt.rc('axes', labelsize=14)    # fontsize of the x and y labels

fig2=plt.figure()
plt.title('using first two PCs $x_{1}$ and $x_{2}$ as features')
#plt.scatter(X_PC12[:15,0],X_PC12[:15,1],c='r',marker='o',label='Apple')
plt.scatter(X_PC12[:,0],X_PC12[:,1],c='y',marker='^',label='review')
plt.legend()
plt.xlabel('$x_{1}$')
plt.ylabel('$x_{2}$')
plt.show()
plt.savefig("img/two-dimensional scatter plot")


# In[22]:


#Categorize the data into two different classes: poor reviews and good reviews. 
#A good review has the star rating above or equal to 3.5. 
#A poor review has a rating of 3.0 or lower. 
#Create a new column, named “category". 
#If the star rating of a comment is 3.0 or less, then categorize the comment as 0 (poor),
#otherwise categorize the comment as 1 (good). 

new_value=np.where(new["stars"]<=3,0,1)
new["categorize"]=new_value
print(new.sample(10))


# In[201]:


# find the 10 most frequent words in one review 
text.max().sort_values(ascending=False)[:10]


# In[202]:


# Find the 10 most frequent words in all review 
word=text.sum(axis=0,skipna=True).sort_values(ascending=False)[:10]
print(word)
fig3=plt.bar(x=word.index,height=word.values)
plt.show()
plt.savefig("img/10 most frequent words")


# In[203]:


##########regression#########
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from IPython.display import display, Math
from sklearn.metrics import r2_score


# In[ ]:


###### The goal of this task is to predict how well the column “useful" characterizes the review text


# In[204]:


# method 1 -use a linear predictor

n=text.shape[1]
x_train=np.array(text[:800])
x_test=np.array(text[800:])
y_train=np.array(new["useful"][:800])
y_test=np.array(new["useful"][800:])

err_train=np.zeros([n,1])
err_val=np.zeros([n,1])

for i in range(n):
    reg=LinearRegression().fit(x_train[:,:(i+1)],y_train)
    
    pre_train=reg.predict(x_train[:,:(i+1)])
    err_train[i]=mean_squared_error(y_train,pre_train)
    
    pre_test=reg.predict(x_test[:,:(i+1)])
    err_val[i]=mean_squared_error(y_test,pre_test)

plt.plot(range(1, n+1), err_train, color='black', label=r'$E_{\rm train}(r)$')
plt.plot(range(1,n+1), err_val, color='red', label=r'$E_{\rm val}(r)$')

plt.title('Training and validation error for different number of features')
plt.ylabel('Empirical error')
plt.xlabel('r features')
plt.xticks(range(1, n + 1))
plt.legend(loc="best")
ax=plt.gca()
ax.set_ylim([0,100])
ax.set_xlim([0,1000])
ax.set_xticks([0,100,200,300,400,500,600,700,800,900,1000])
plt.show()
    


# In[ ]:





# In[205]:


plt.plot(range(1, 300), err_train[1:300], color='black', label=r'$E_{\rm train}(r)$',linewidth=2)
plt.plot(range(1,300), err_val[1:300], color='red', label=r'$E_{\rm val}(r)$', marker='x',linewidth=2)

plt.title('Training and validation error for different number of features')
plt.ylabel('Empirical error')
plt.xlabel('r features')
plt.xticks(range(1, n + 1))
plt.legend(loc="best")
ax=plt.gca()
ax.set_ylim([0,30])
ax.set_xlim([0,300])
ax.set_xticks([0,50,100,150,200,250,300])
plt.show()


# In[206]:


# method 2- Huber regression
# varying number of feature with Huber loss
from sklearn import linear_model
from sklearn.linear_model import HuberRegressor

n=text.shape[1]

err_train_hub=np.zeros([n,1])
err_val_hub=np.zeros([n,1])

for i in range(n):
    reg_hub=HuberRegressor().fit(x_train[:,:(i+1)],y_train)
    
    pre_train=reg_hub.predict(x_train[:,:(i+1)])
    err_train_hub[i]=mean_squared_error(y_train,pre_train)
    
    pre_test=reg_hub.predict(x_test[:,:(i+1)])
    err_val_hub[i]=mean_squared_error(y_test,pre_test)

plt.plot(range(1, n+1), err_train_hub, color='black', label=r'$E_{\rm train}(r)$')
plt.plot(range(1,n+1), err_val_hub, color='red', label=r'$E_{\rm val}(r)$')

plt.title('Training and validation error for different number of features')
plt.ylabel('Empirical error')
plt.xlabel('r features')
plt.xticks(range(1, n + 1))
plt.legend(loc="best")
ax=plt.gca()
ax.set_ylim([0,30])
ax.set_xlim([0,1000])
ax.set_xticks([0,100,200,300,400,500,600,700,800,900,1000])
plt.show()
    


# In[207]:


## standized  the dataset
sta_text=StandardScaler().fit_transform(text)
y=np.array(data["useful"]).reshape(-1,1)
#sta_useful=StandardScaler().fit_transform(useful)


# In[34]:


### apply PCA, plot the variance explained by components and determine the number of components

pca = PCA().fit(sta_text)
var=np.cumsum(pca.explained_variance_ratio_)
fig1=plt.plot(var)
plt.xlabel("number of components")
plt.ylabel("cumulative explained variance")
plt.title("cumulative explained variance based on number of components")
plt.show()
plt.savefig("img/cumulative explained variance of all data")


# In[8]:


## apply PCA with Huberregressor

# can change the D
D= 200
z_train=np.array(sta_text[:800])
z_val=np.array(sta_text[800:])
y_train=np.array(y[:800])
y_val=np.array(y[800:])


err_train=np.zeros(D)
err_val=np.zeros(D)
r2_train=np.zeros(D)
r2_val=np.zeros(D)

for n in range(1,D+1):    
    # Create the PCA object and fit
    pca=PCA(n_components=n)
    pca.fit(sta_text)

    # transform long feature vectors (length D) to short ones (length n)
    x_train=pca.transform(z_train)
    x_val=pca.transform(z_val)
    
    # use x_train, x_value to train linear regression model
    reg=HuberRegressor()
    reg=reg.fit(x_train,y_train)
    y_pred_train=reg.predict(x_train)
    y_pred_val=reg.predict(x_val)

    err_train[n-1] = mean_squared_error(y_train,y_pred_train)  # compute training error 
    err_val[n-1] = mean_squared_error(y_val,y_pred_val)   # compute validation error 
    
    r2_train[n-1]=r2_score(y_train,y_pred_train) 
    r2_val[n-1]=r2_score(y_val,y_pred_val)  


# In[83]:


plt.plot(range(1,D+1,1),err_val,label="validation")
plt.plot(range(1,D+1,1),err_train,label="training")
plt.xlabel('number of PCs ($n$)')
plt.ylabel(r'error')
plt.legend()
plt.title('validation/training error vs. number of PCs')
ax=plt.gca()
ax.set_ylim([0,20])
ax.set_xlim([0,200])
ax.set_xticks([0,25,50,75,100,125,150,175,200])
plt.show()


# In[89]:


plt.plot(range(1,D+1,1),r2_val,label="validation")
plt.plot(range(1,D+1,1),r2_train,label="training")
plt.xlabel('number of PCs ($n$)')
plt.ylabel(r'R2 score')
plt.legend()
plt.title('validation/training R2 socre vs. number of PCs')
ax=plt.gca()
ax.set_ylim([0,0.3])
ax.set_xlim([0,200])
ax.set_xticks([0,25,50,75,100,125,150,175,200])
plt.show()


# In[58]:


#### classification- Logistic Regression CV
### predict the label (good or poor) of a review based on the review text

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

x_train=np.array(text[:800])
x_val=np.array(text[800:])
y_train=new["categorize"][:800]
y_val=new["categorize"][800:]


clf=LogisticRegression(random_state=0).fit(x_train,y_train)
pred_y_val=clf.predict(x_val)

cm=confusion_matrix(y_val,pred_y_val)
print(cm)

labels=["good","poor"]
fig5=plt.figure()
ax=fig5.add_subplot()
sns.heatmap(cm,cmap="YlGnBu")

plt.title("confusion matrix of the classifier")
plt.xlabel("predicted")
plt.ylabel("true")
ax.xaxis.set_ticklabels(['poor', 'good'])
ax.yaxis.set_ticklabels(['poor', 'good'])


# In[17]:


#### Clustering-k-means clustering
# cluster the review text data into different groups and generate the word cloud maps.

from sklearn.cluster import KMeans

X=np.zeros([1000,1080])
X=np.array(text)

# try out different amounts of clusters(10-15), determine the reasonal one
data_num=X.shape[0]
err_clustering=np.zeros([15,1])

for k in range(15):
    k_means=KMeans(n_clusters=k+1, max_iter=100).fit(X)
    err_clustering[k]=k_means.inertia_/data_num


# In[18]:


fig=plt.figure(figsize=(8,6))
plt.plot(range(1,16),err_clustering)
plt.xlabel('Number of clusters')
plt.ylabel('Clustering error')
plt.title("The number of clusters vs clustering error")
ax=plt.gca()

ax.set_xlim([0,16])
plt.show()    


# In[96]:


#### repeat k-means 20 times using k=12 clusters and L=100 interations in each repetition.
min_ind=0

X=np.zeros([1000,1080])
X=np.array(text)

cluster_assignment=np.zeros((50,X.shape[0]),dtype=np.int32)
clustering_err=np.zeros([50,1])

np.random.seed(42)
init_means_cluster1 = np.random.randn(50,1080)  # use the rows of this numpy array to init k-means 
init_means_cluster2 = np.random.randn(50,1080)  # use the rows of this numpy array to init k-means 
init_means_cluster3 = np.random.randn(50,1080)  # use the rows of this numpy array to init k-means 
init_means_cluster4 = np.random.randn(50,1080)  # use the rows of this numpy array to init k-means 
init_means_cluster5 = np.random.randn(50,1080)  # use the rows of this numpy array to init k-means 
init_means_cluster6 = np.random.randn(50,1080)  # use the rows of this numpy array to init k-means 
init_means_cluster7 = np.random.randn(50,1080)  # use the rows of this numpy array to init k-means 
init_means_cluster8 = np.random.randn(50,1080)  # use the rows of this numpy array to init k-means 
init_means_cluster9 = np.random.randn(50,1080)  # use the rows of this numpy array to init k-means 
init_means_cluster10 = np.random.randn(50,1080)  # use the rows of this numpy array to init k-means 
init_means_cluster11 = np.random.randn(50,1080)  # use the rows of this numpy array to init k-means 
init_means_cluster12 = np.random.randn(50,1080)  # use the rows of this numpy array to init k-means 

best_perform=np.zeros((1000,1))

init_means_cluster_ar=np.zeros((50,12,1080))
for i in range(50):
    init_means_cluster_ar[i]=[init_means_cluster1[i,:],
                              init_means_cluster2[i,:],
                              init_means_cluster3[i,:],
                              init_means_cluster4[i,:],
                              init_means_cluster5[i,:],
                              init_means_cluster6[i,:],
                              init_means_cluster7[i,:],
                              init_means_cluster8[i,:],
                              init_means_cluster9[i,:],
                              init_means_cluster10[i,:],
                              init_means_cluster11[i,:],
                              init_means_cluster12[i,:]]
    
data_num = X.shape[0]

for i in range(50):
    k_means=KMeans(n_clusters = 12,init=init_means_cluster_ar[i],max_iter = 100).fit(X)
    err_clustering=k_means.inertia_/data_num
    clustering_err[i]=err_clustering
    cluster_assignment[i]=k_means.labels_

min_ind=np.argmin(clustering_err)
best_perform=cluster_assignment[min_ind]


# In[101]:


fig=plt.figure(figsize=(8,6))
plt.plot(range(1,51),clustering_err)
plt.xlabel('Number of repetition')
plt.ylabel('Clustering error')
plt.title("The number of repetition vs clustering error")
ax=plt.gca()
#ax.set_xticks([2,4,6,8,10,12,14,16,18,20])
plt.show() 


# In[124]:


plt.hist(best_perform)
plt.show()


# In[105]:


text.loc[best_perform==6]


# In[131]:


#plot the wordcloud for each cluster
def plot_wordcloud (index):
    s_text=text.loc[best_perform==index]
    w_word=s_text.sum(axis=0,skipna=True).sort_values(ascending=False)[:100]
    wordcloud=WordCloud(background_color="white").generate(str(list(w_word.index)))
    plt.figure()
    plt.title("wordcloud of cluster {}".format(index))
    plt.imshow(wordcloud,interpolation="bilinear")
    plt.axis("off")
    plt.show()
    wordcloud.to_file("img/wordcloud of cluster{}.png".format(index))
    


# In[132]:


def label (index):
    rating_stars=np.mean(new.loc[best_perform==index]["stars"])
#    rating_cool=np.mean(new.loc[best_perform==index]["cool"])
#    rating_funny=np.mean(new.loc[best_perform==index]["funny"])
#    rating_useful=np.mean(new.loc[best_perform==index]["useful"])
    print("rating_stars",rating_stars)


# In[133]:


for i in range(12):
    plot_wordcloud(i)
    label(i)


# In[21]:


def cluster_plots(set1, colours1,
                  title1 = 'Dataset 1'):
    colours1 = colours1.reshape(-1,)
    fig= plt.subplots(111)
    ax.scatter(set1[:, 0], set1[:, 1],s=8,lw=0,c=colours1)
    ax.set_title(title1,fontsize=14)
    ax.set_xlim(min(set1[:,0]), max(set1[:,0]))
    ax.set_ylim(min(set1[:,1]), max(set1[:,1]))
    
    plt.show()


# In[19]:


### Clustering with DBSCAN.
from sklearn.cluster import DBSCAN

dbscan=DBSCAN(eps=1,min_samples=10,metric='euclidean').fit_predict(X).reshape(-1,1)
dbscan_dataset1_sel=np.select([dbscan==-1],[1],default=0)
dataset1_noise_points=int(sum(dbscan_dataset1_sel))


# In[20]:





# In[ ]:




