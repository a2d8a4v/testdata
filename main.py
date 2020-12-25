#!/usr/bin/env python
# coding: utf-8

## import modules
import glob , os , re , random , sys
import pickle , csv
import numpy as np
import collections
from tqdm import tqdm
from datasets import load_dataset


# ending_names = [ f"ending{i}" for i in range(4) ]

# context_name = "doc_id"
# question_header_name = "doc_text"

# documents = load_dataset('csv', data_files=prefix + "documents.csv" )
# print(documents)
# dataset = documents['train']
# print(dataset[:1])

# first_sentences = [[context] * 4 for context in dataset]
# question_headers = dataset[question_header_name]
# second_sentences = [
#     [f"{header} {dataset[end][i]}" for end in ending_names] for i, header in enumerate(question_headers)
# ]

# # Flatten out
# first_sentences = sum(first_sentences, [])
# second_sentences = sum(second_sentences, [])

# print( first_sentences )


def pickleStore( savethings , filename ):
    dbfile = open( filename , 'wb' )
    pickle.dump( savethings , dbfile )
    dbfile.close()
    return


def pikleOpen( filename ):
    file_to_read = open( filename , "rb" )
    p = pickle.load( file_to_read )
    return p


if __name__  == "__main__":

    ## settings
    csv.field_size_limit( sys.maxsize )

    ## global variables
    dic_sources = 'ntust-ir2020-homework6/'
    dic_save    = dic_sources + 'save/'
    # create directory
    if not os.path.exists( dic_save ):
        os.makedirs( dic_save )

    ## preprocessing
    # save document file
    docs_dict = {}
    with open( dic_sources + 'documents.csv' , newline='' ) as csvfile:
        spamreader = csv.reader( csvfile , delimiter=',' )
        for c , row in enumerate( spamreader ):
            # get content without first line features name
            if c > 0:
                string = row[1].split( "[Text]" )
                if len( string ) == 1:
                    docs_dict[ row[0] ] =  " ".join( re.sub( r'\W+' , ' ' , string[0] ).replace( "\n" , " " ).split() )
                else:
                    docs_dict[ row[0] ] =  " ".join( re.sub( r'\W+' , ' ' , string[1] ).replace( "\n" , " " ).split() )
    pickleStore( docs_dict , dic_save + "docs_dict.pkl" )

    # make first sentence
    queries_dict = {}
    que_pos_dict = {}
    que_top_dict = {}
    with open( dic_sources + 'train_queries.csv' , newline='' ) as csvfile:
        spamreader = csv.reader( csvfile , delimiter=',' )
        for c , row in enumerate( spamreader ):
            if c > 0:
                queries_dict[ row[0] ] = " ".join( re.sub( r'\W+' , ' ' , row[1] ).split() )
                que_pos_dict[ row[0] ] = row[2]
                que_top_dict[ row[0] ] = dict( zip( row[3].split() , row[4].split() ) )
    pickleStore( queries_dict , dic_save + "queries_dict.pkl" )
    pickleStore( que_pos_dict , dic_save + "que_pos_dict.pkl" )
    pickleStore( que_top_dict , dic_save + "que_top_dict.pkl" )

    # make first sentence
    tmp = 0
    # first_sentences = [ [ context ] * 4 for context in dataset ]
    for query_name , query_content in queries_dict.items():
        with open( dic_save + "train.csv" , "a" ) as writefile:
            if tmp == 0:
                writefile.write( "query_name,query_content,positive,negative,label\n" )
                tmp += 1
            positive_list = que_pos_dict[ query_name ].split()
            for i in range( len( positive_list ) ):
                positive = positive_list[ i ]
                negative = " ".join( random.choices( [ x for x in que_top_dict[ query_name ].keys() if x not in que_pos_dict[ query_name ].split() ] , k=3 ) )
                append = ",".join( [ query_name , query_content , positive , negative , str(1) ] )
                writefile.write( append + "\n" )

