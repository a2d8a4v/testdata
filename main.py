#!/usr/bin/env python
# coding: utf-8

## import modules
import glob , os , re , random , sys , math
import pickle , csv
import numpy as np
import collections
from tqdm import tqdm
from datasets import load_dataset


def pickleStore( savethings , filename ):
    dbfile = open( filename , 'wb' )
    pickle.dump( savethings , dbfile )
    dbfile.close()
    return


def pikleOpen( filename ):
    file_to_read = open( filename , "rb" )
    p = pickle.load( file_to_read )
    return p


def shuffleCutList( big_lists ):
    new = []
    times = math.ceil( len( big_lists ) / 4 )
    for t in range( times ):
        if len( big_lists ) >= 4:
            tmp_l = random.choices( big_lists , k=4 )
        else:
            tmp_l = big_lists
        new.append( tmp_l )
        big_lists = [ x for x in big_lists if x not in tmp_l ]
    return new


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

    # make train data
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
    pickleStore( queries_dict , dic_save + "train_queries_dict.pkl" )
    pickleStore( que_pos_dict , dic_save + "train_que_pos_dict.pkl" )
    pickleStore( que_top_dict , dic_save + "train_que_top_dict.pkl" )
    tmp = 0
    for query_name , query_content in queries_dict.items():
        with open( dic_save + "all.csv" , "a" ) as writefile:
            if tmp == 0:
                writefile.write( "query_name,query_content,answer,label\n" )
                tmp += 1
            positive_list = que_pos_dict[ query_name ].split()
            for i in range( len( positive_list ) ):
                positive = positive_list[ i ]
                # negative = " ".join( random.choices( [ x for x in que_top_dict[ query_name ].keys() if x not in que_pos_dict[ query_name ].split() ] , k=3 ) )
                negative = random.choices( [ x for x in que_top_dict[ query_name ].keys() if x not in que_pos_dict[ query_name ].split() ] , k=3 )
                # print( len( que_pos_dict[ query_name ].split() ) , len( [ x for x in que_top_dict[ query_name ].keys() if x not in que_pos_dict[ query_name ].split() ] ) )
                random_l = [ positive ] + negative
                random.shuffle( random_l )
                index_el = random_l.index( positive )
                random_l = " ".join( random_l )
                append = ",".join( [ query_name , query_content , random_l , str( index_el ) ] )
                writefile.write( append + "\n" )

    datasets = load_dataset( "csv" , data_files=dic_save + "all.csv" )
    datasets = datasets['train'].train_test_split( test_size=0.1 , shuffle=True ) 
    tmp = 0
    for row in datasets['train']:
        with open( dic_save + "train.csv" , "a" ) as writefile:
            if tmp == 0:
                writefile.write( "query_name,query_content,answer,label\n" )
                tmp += 1
            append = ",".join( [ str(x) for x in row.values() ] )
            writefile.write( append + "\n" )

    tmp = 0
    for row in datasets['test']:
        with open( dic_save + "validation.csv" , "a" ) as writefile:
            if tmp == 0:
                writefile.write( "query_name,query_content,answer,label\n" )
                tmp += 1
            append = ",".join( [ str(x) for x in row.values() ] )
            writefile.write( append + "\n" )


    # make test data
    queries_dict = {}
    que_top_dict = {}
    que_all_list = []
    with open( dic_sources + 'test_queries.csv' , newline='' ) as csvfile:
        spamreader = csv.reader( csvfile , delimiter=',' )
        for c , row in enumerate( spamreader ):
            if c > 0:
                queries_dict[ row[0] ] = " ".join( re.sub( r'\W+' , ' ' , row[1] ).split() )
                que_top_dict[ row[0] ] = dict( zip( row[2].split() , row[3].split() ) )
    pickleStore( queries_dict , dic_save + "test_queries_dict.pkl" )
    pickleStore( que_top_dict , dic_save + "test_que_top_dict.pkl" )
    for query_name , query_content in queries_dict.items():
        docs_list = list( que_top_dict[ query_name ].keys() )
        doc4_list = shuffleCutList( docs_list )
        for x in doc4_list:
            que_all_list.append( ",".join( [ query_name , query_content , " ".join( x ) ] ) )
    random.shuffle( que_all_list )
    with open( dic_save + "test.csv" , "a" ) as writefile:
        writefile.write( "query_name,query_content,top1000\n" )
        for row in que_all_list:
            writefile.write( row + "\n" )
