from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import re
import gensim
import pickle
import json
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from nltk.tokenize import word_tokenize
from os import listdir
from os.path import isfile, join
from gensim.models.deprecated.doc2vec import LabeledSentence


stopword_set = set(stopwords.words('english'))


app = Flask(__name__)

@app.route('/')
def my_form():
    return render_template('my-form.html')

@app.route('/', methods=['POST'])
def my_form_post():
	search = request.form['text']

	with open('abmodel.pkl', 'rb') as pkl:
		abmodel = pickle.load(pkl)
	with open('abmodel_t.pkl', 'rb') as pkl:
		abmodel_t = pickle.load(pkl)
	with open('doubleTowns.pkl', 'rb') as pkl:
		doubleTowns = pickle.load(pkl)
	with open('doubleCountry.pkl', 'rb') as pkl:
		doubleCountry = pickle.load(pkl)
	with open('df_infoLink.pkl', 'rb') as pkl:
		df_infoLink = pickle.load(pkl)


	query = search.lower().strip("!.?").replace(',','')
	query = query.split(' ')
	q_temp = query
	rQuery  = [q_temp[0]]

	for i in range(1, len(q_temp)):
		if (q_temp[i-1] + '-' + q_temp[i]) in doubleTowns:
			rQuery.pop()
			rQuery.append(q_temp[i-1] + '-' + q_temp[i])
		elif (q_temp[i-1] + '-' + q_temp[i]) in doubleCountry:
			rQuery.pop()
			rQuery.append(q_temp[i-1] + '-' + q_temp[i])
		elif (q_temp[i-1] + '-' + q_temp[i]) == 'south-america' or (q_temp[i-1] + '-' + q_temp[i]) == 'north-america' or (q_temp[i-1] + '-' + q_temp[i]) == 'middle-east':
			rQuery.pop()
			rQuery.append(q_temp[i-1] + '-' + q_temp[i])
		else:
			rQuery.append(q_temp[i])

	query = ' '.join(word for word in rQuery)

	query = word_tokenize(query)
	query = list(set(query).difference(stopword_set))

	q_vec = abmodel.infer_vector(query)
	results = abmodel.docvecs.most_similar(positive=[q_vec], topn = 100000)

	q_vec_t = abmodel_t.infer_vector(query)
	results_t = abmodel_t.docvecs.most_similar(positive=[q_vec_t], topn = 100000)

	df_results = pd.DataFrame(results, columns=['title','similarity'])
	df_results_t = pd.DataFrame(results_t, columns=['title','similarity'])

	def isCountry(x):
	    
	    country = x.split('_')[1]
	    theQ = query
	    
	    if country in theQ:
	        return 1
	    else:
	        return 0
	    
	def isTown(x):
	    
	    town = x.split('_')[2]
	    theQ = query
	    
	    if town in theQ:
	        return 1
	    else:
	        return 0
	    
	def isContinent(x):
	    
	    continent = x.split('_')[3]
	    theQ = query
	    
	    if continent in theQ:
	        return 1
	    else:
	        return 0

	#Break up the title piece
	df_results['firstParagraph'] = df_results['title'].apply(lambda x: 1 if x.split('_')[0][-2:] == '00' else 0)
	df_results['isCountry'] = list(map(isCountry, df_results['title']))
	df_results['isTown'] = list(map(isTown, df_results['title']))
	df_results['isContinent'] = list(map(isContinent, df_results['title']))

	df_results_t['firstParagraph'] = df_results_t['title'].apply(lambda x: 1 if x.split('_')[0][-2:] == '00' else 0)
	df_results_t['isCountry'] = list(map(isCountry, df_results_t['title']))
	df_results_t['isTown'] = list(map(isTown, df_results_t['title']))
	df_results_t['isContinent'] = list(map(isContinent, df_results_t['title']))

	#reset title column and sort values to get the results sorted
	df_results['title'] = df_results['title'].apply(lambda x: x.split('_')[0][:-2])
	df_results_t['title'] = df_results_t['title'].apply(lambda x: x.split('_')[0][:-2])

	#Filter for results containing a city or country
	def getResults(df):
	    df_temp1 = df[df['isTown'] == 1]
	    if len(df_temp1) == 0:
	        df_temp2 = df[df['isCountry'] == 1]
	        if len(df_temp2) == 0:
	            df_temp3 = df[df['isContinent'] == 1]
	            if len(df_temp3) == 0:
	                return df
	            else:
	                return df_temp3
	        else:
	            return df_temp2
	    else:
	        return df_temp1
	    
	df_results = getResults(df_results)
	df_results_t = getResults(df_results_t)

	#Break up between first paragraph similarities and non-first
	df_results_firstP = df_results[df_results['firstParagraph'] == 1]
	df_results = df_results[df_results['firstParagraph'] == 0]

	#Get the top similarity per article 
	df_results = df_results.sort_values(['title', 'similarity'], ascending=[True, False])
	df_results_big = df_results.drop_duplicates('title', keep='first')


	def makeCombo(x, y):
	    return str(x) + '_' + str(y)

	df_results['combo'] = list(map(makeCombo, df_results['title'], df_results['similarity']))
	df_results_big['combo'] = list(map(makeCombo, df_results_big['title'], df_results_big['similarity']))

	naughty = df_results_big['combo'].tolist()

	def makeTable(x):
	    
	    
	    if x in naughty:
	        return 1
	    else:
	        return 0
	    
	df_results['combo_link'] = list(map(makeTable, df_results['combo']))

	df_results = df_results[df_results['combo_link'] == 0]


	#Body article scores
	body = df_results.groupby('title').mean().sort_values(by='similarity', ascending=False).reset_index()
	body = body.drop(['firstParagraph','isCountry','isTown','isContinent','combo_link'], axis=1)
	body = body.rename(index=str, columns={'similarity':'similarity_body'})
	body['similarity_body'] = body['similarity_body'].apply(lambda x: x*0.27)

	#First Paragraph scores
	firstP = df_results_firstP.groupby('title').mean().sort_values(by='similarity', ascending=False).reset_index()
	firstP = firstP.drop(['firstParagraph','isCountry','isTown','isContinent'], axis=1)
	firstP = firstP.rename(index=str, columns={'similarity':'similarity_firstP'})
	firstP['similarity_firstP'] = firstP['similarity_firstP'].apply(lambda x: x*0.70)

	#Title scores
	title = df_results_t.groupby('title').mean().sort_values(by='similarity', ascending=False).reset_index()
	title = title.drop(['firstParagraph','isCountry','isTown','isContinent'], axis=1)
	title = title.rename(index=str, columns={'similarity':'similarity_title'})
	title['similarity_title'] = title['similarity_title'].apply(lambda x: x*0.03)

	final_temp = pd.merge(firstP, body, how='left', on='title')
	df_final = pd.merge(final_temp, title, how='left', on='title')

	def getFinalScore(x,y,z):
	    return x+y+z

	df_final['final_score'] = list(map(getFinalScore, df_final['similarity_body'],df_final['similarity_firstP'], df_final['similarity_title']))
	df_finalFinal = pd.merge(df_final, df_infoLink, how='left', on='title')
	df_final = df_finalFinal.sort_values(by='final_score', ascending=False)

	x = df_final[['title','URL','image_one']].iloc[0:3]

	y = list(zip(x.title, x.URL, x.image_one))

	

	return render_template("result.html", result=y)

if __name__ == "__main__":
	app.run(debug=True)