# Social topics throughout music: sentiment analysis
#### Abstract
In this project, our goal is to find out about the social topics evolution in culture. We will use music lyrics from the “Million song dataset” to analyse sentiment in tracks from artist, genres and time. What can the sentiment towards different topics in songs tell us about people and the time they live in? Our goal is to provide a visualization of semantic trends in music, that hopefully will give people more insight into everyday social topics and how they evolve as a part of our culture.
To visualize our data, we want high interactivity so one can have filter by one’s own interest in this high dimensional space. Therefore, our work will be presented in the form of an interactive blog. 

## Milestone 2
To achieve this task, we divided our work in 4 parts:


### 1. Data imports: structures, sorting and wrangling

Importing the data from the sets that we need to have, handling the data in its size. Because our dataset is quite small in terms of volume, we are able to store it directly on our machines and deal with it easily. See the .ipynb file for the details of the data wrangling

### 2. Classifiers: Choosing the methods of analysis and extracting features
We examine various features, they are the following:
- Complexity of the lyrics
- Polarity of the lyrics
- Evolution of complexity through time inside a genre

### 3. Visualization: How to meaningfully represent the data
We provide a data strcture that enables straightforward visualization. We present what we think is the most fitting way to visualize our data. 

Structures of interactive visualizations are also presented in the notebook. 

### 4. Decisions: What can be said from the outputs, and what is left to do
#### 4.1. Intermediary conclusions
##### Data handling
Due to the fact that we are solely treating text, and that the lyrics are not entirely given, we use a volume of data that is relatively small. This situation enables us to have the original text files stored on our machines, and proceeding to storage of value when using the notebook is not necessary since our computer's cache is big enough to treat our data.

##### Visualization
We are convinced that the barplot is the most fitting representation for the generated graphs. We might find another librairy that makes it look more visually appealing, but we believe that it is the most meaningful way to show our results.

##### Comparison
Once we will have a topic-based split of our data, we will be able to compare how topics are treated in different genres. This will help us to establish how social topics are treated throughout music genres. Interactive visualization will help users to compare the different features. 

#### 4.2. Further objectives

- Find a classification to have topics, and if no useful algorithm, use the one we built on the 20newsgroup.
Establish a host and astructure for a blog. Now that the raw material is present, we must seek for a layout that fits our project. 
- Expand our analysis to the whole dataset. As discussed earlier, we only treated a sample for this milestone, in order to show what we are aiming at. 

##### 4.3. Planning until milestone 3

04.12. 
- Decide which option we should consider for the topic analysis (if nothing relevant is find for option 2, we will use option 1).
- Generalize our data treatment to the whole dataset.
- Generate the visualizations we want to have. 
- Decide what type of storage is the most efficient for our generated graphics
- Start build the blog


11.12.
- Finish the written description (data story)
- Debug our code
- Debug the implementation of the blog

19.12. 
- Handout milestone 3


## Questions for the TA
- Is there a deep learning that could classify our bags of words by topic? We believe that something implement with TF-IDF as a starting point would be useful but could not find anything concluant yet.
- Is there a blog platform that could host our data structure and visualization for free?
- Does it seem to you that the analysis we are aiming at is meaningful and relevant enough to be considered as "Handling data for social good"?







## Milestone 1

### Research questions
We will guide our work with the following questions:

- How is the perception of certain topics evolving throughout time and within genres?
- How is the relationship between topics and locations characterized in different music genres?
- Is there a correlation between the complexity of lyrics and specific topics? 

### Dataset
Our main dataset is coming from the one called “Million song” that was provided with the project list. As specified in the abstract we will be working on the lyrics with the “musiXmatch” dataset and the “tagtraum” dataset for genres. We might use the “Top MAGD” dataset that gives more music genres, it depends on how performant we manage to be with the first genre dataset. We will also use NLTK to analyse whether the words/sentences that are used are of a positive or negative nature. To get additional specific information from artists, like pictures and origin, we will use Wikidata 

### A list of internal milestones up until project milestone 2
04.11

- Download the required data
- Set up the Git and project skeleton

11.11

- Establish the sentiment analysis on all the song of our dataset and create a new table
- Append an additional rating to the positive/negative one that is provided by NLTK
- Find the most powerful way to store and look up all our data
- Find how we can clean our dataset

19.11

- Explore possibilities for the data visualization
- Decide which visualization format is the best.
- Characterization of the artists and lyrics in genres and grouping
- Use the data from WikiData to get extra information such as pictures or precise artists’ origins

26.11

- Comment and debug our code
- Set up our goals and plans for the next milestone.

##### Options to consider:

- Extend our work to others languages than English

### Questions for TAa
- Are we allowed to use pictures for representation of artists if our work is not publicly published? Copyright-wise, how does this work?
- Is the content we have enough to make a project? 
- Is the idea fitting to the “social good” topic?
- In terms of visualization and analysis, which aspect is the most important one?
