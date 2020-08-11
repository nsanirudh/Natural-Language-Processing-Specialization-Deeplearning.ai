### NLP Specialization 

# 2. Natural Language Processing with Probabilistic Models

# Notes made on Typora

## Week 1

Overview

>Auto-correct
>
>Build the model
>
>Minimum edit distance
>
>Dynamic Programming

####  Auto correct

* What is autocorrect?
  * Corrects a sentence with a mistake (deah or deer -> dear)
* Steps in algorithm
  * Identify the misspelled word
  * find strings n-edit distances away
  * filter candidates
  * calculate word probabilities 

#### Build the model

* if a word is misspelled it is identified by checking if it is part of a dictionary(vocabulary) 
* Find words n-edit distance away from the identified word
  * an Edit is an operation performed on a string to change it, like:
  * insert
  * delete
  * swap
  * replace
* Filter the candidates of the edited, identified string  by checking them in your vocabulary
* Calculate word probabilities, which is the occurrence of word divided by the total number of words in the vocabulary
* The word with with the highest probability is the candidate for replacement of the identified misspelled word

#### Minimum Edit Distance

Is the minimum number of edits required to transform one string into another

* cost of insert operation -1 
* cost of delete operation - 1
* cost of replace operation - 2

#### Dynamic programming

* Given a source word and a target word
* We make a table so that we can transform the word with the the minimum cost
* Based on Levensthein distance

## Week 2

