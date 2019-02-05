# Hybrid Recommender

This package usage multiple algorithms and parameters to accomodate different set of use cases.

### Parameters:
* **item_clusters**: int
    The number of clusters for item matrix generation. This parameter can be tuned
* **top_results**: int
    Number of recommendations needed. Default value is 10
* **ratings_weightage**: int
    Weightage for user ratings score. Default is 1
* **content_weightage**: int
    Weightage for content score. Default is 1
* **null_rating_replace**: str
    Value to be used as replacement for missing ratings. Default is 'mean', other acceptable values are 'zero','one', and 'min'

### Returns:
    DataFrame having top recommended results for the list of users

### Approach:

1. Create an instance of the hybrid recommender class
     mr = hybrid_recommender()

2. Call fit method on the defined object by passing on ratings and content data
     mr.fit(ratings_df,content_df)

3. Call the predict method
    recommended_df = mr.predict()

------------------------------------------------------------

       
## Example

### Create Ratings DataFrame
```python
item_id = [1,7,9,10,12,2,4,6,8,10,12,3,6,9,12,14,10,13,12,14,11,2,5,7,8,9,10,12]
user_id = [1,1,1,1,1,2,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4,5,5,5,5,5,5,5]
rating = [4,5,2,3,5,2,3,2,3,4,4,5,1,2,3,1,2,4,5,3,5,3,1,3,5,3,5,3]
ratings = pd.DataFrame({'user_id':user_id,'item_id':item_id,'rating':rating})
```
### Create Content DataFrame
```python
items = [1,2,3,4,5,6,7,8,9,10,11,12,13,14]
cols = ['col1','col2','col3','col4','col5']
feats =[[1,0,0,1,1],
       [1,1,0,0,1],
       [0,1,1,0,0],
       [0,1,1,1,0],
       [1,0,1,1,1],
       [1,1,1,0,0],
       [0,1,0,1,0],
       [0,0,0,1,0],
       [0,1,1,0,0],
       [1,1,1,0,1],
       [0,0,0,1,1],
       [0,1,0,1,0],
       [0,1,1,0,1],
       [0,0,1,1,1],]
item_df = pd.DataFrame(feats,index=items,columns=cols)
```
**Ratings DataFrame**
```python
ratings.head()
```
**Content DataFrame**
```python
item_df.head()
```
### Fitting and prediction

**Creating the recommender object**
```python
my_recommender = hybrid_recommenders(item_clusters=4,top_results=5)
```
**Fitting the data**
```python
my_recommender.fit(ratings,item_df)
```
**Recommend for few users**
```python
my_recommender.predict([1,2,3])
```
**Recommendations for All users**
```python
my_recommender.predict()
```
