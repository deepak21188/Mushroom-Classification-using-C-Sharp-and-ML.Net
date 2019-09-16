# Mushroom Classifier Using C# and ML.Net
| ML Task        | Screnario        | Platform  | Langauge  | ML.NET version| Algorithms| Data type| App type |
| ------------- | ------------- | ----- | ------ |--------------|----------| ------ | ------ |
| Binary Classification      | Classifiy Mushrooms(Edible/poisonous) | Dot Net Core 2.1|C# | V 1.3.1 | AveragedPerceptron and OneVersusAll | csv file| Console app |

### Problem
This project demostrat the application of ML.Net to classify the mushrooms whether they are Edible or poisonous. This type of task is very popular in machine learning world and is often referred as Two-class or Binary classification problem. The pupose of this project is to see how can we leverage the wonderful capabilities of ML.Net to implement machine learning based features into our .net applications.

### Data

This dataset includes descriptions of hypothetical samples corresponding to 23 species of gilled mushrooms in the Agaricus and Lepiota Family Mushroom drawn from The Audubon Society Field Guide to North American Mushrooms (1981). Each species is identified as definitely edible, definitely poisonous, or of unknown edibility and not recommended. This latter class was combined with the poisonous one. The Guide clearly states that there is no simple rule for determining the edibility of a mushroom; no rule like "leaflets three, let it be'' for Poisonous Oak and Ivy.

#### Attribute Information
##### Featues
|cap-shape|cap-surface|cap-color|bruises|odor|gill-attachment|gill-spacing|gill-size|gill-color|stalk-shape|stalk-root|stalk-surface-above-ring|stalk-surface-below-ring|stalk-color-above-ring|stalk-color-below-ring|veil-type|veil-color|ring-number|ring-type|spore-print-color|population|habitat|
|---------|-----------|---------|-------|----|---------------|------------|---------|----------|-----------|----------|------------------------|------------------------|----------------------|----------------------|---------|----------|-----------|---------|-----------------|----------|-------|
|bell=b,conical=c,convex=x,flat=f, knobbed=k,sunken=s|fibrous=f,grooves=g,scaly=y,smooth=s|brown=n,buff=b,cinnamon=c,gray=g,green=r,pink=p,purple=u,red=e,white=w,yellow=y|bruises=t,no=f|almond=a,anise=l,creosote=c,fishy=y,foul=f,musty=m,none=n,pungent=p,spicy=s| attached=a,descending=d,free=f,notched=n |close=c,crowded=w,distant=d|broad=b,narrow=n| black=k,brown=n,buff=b,chocolate=h,gray=g, green=r,orange=o,pink=p,purple=u,red=e,white=w,yellow=y|enlarging=e,tapering=t|bulbous=b,club=c,cup=u,equal=e,rhizomorphs=z,rooted=r,missing=?|fibrous=f,scaly=y,silky=k,smooth=s|fibrous=f,scaly=y,silky=k,smooth=s|brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y|brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y|partial=p,universal=u|brown=n,orange=o,white=w,yellow=y|none=n,one=o,two=t|cobwebby=c,evanescent=e,flaring=f,large=l,none=n,pendant=p,sheathing=s,zone=z|black=k,brown=n,buff=b,chocolate=h,green=r,orange=o,purple=u,white=w,yellow=y|abundant=a,clustered=c,numerous=n,scattered=s,several=v,solitary=y|grasses=g,leaves=l,meadows=m,paths=p,urban=u,waste=w,woods=d


##### Label (Class)
|Label|
|-----|
|edible=e, poisonous=p|

### Solution

To solve this problem we peform the following steps

#### Step 1: 
* Load the dataset from csv data file
* Divide it into training and testing sets in the ratio of 75:25

#### Step 2:
* Preprocess the data - Create an estimator and transform the data to the numeric vectrors so that it can be used by ML algorithm.
* Choose and append the classification algorithm to the pipeline

#### Step 3:
* Crossvalidate the model to check its performance. Crossvalidation provide model's performance based on vairous evaluation metrics.

#### Step 4:
* Train the model by providing training dataset as input to the model.

#### Step 5:
* Once the model is trained with training data, it can be used/consumed to predict the classes/Labels of new data samples.



 



