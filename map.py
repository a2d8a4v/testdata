#!/usr/bin/env python
# coding: utf-8

import csv
import numpy as np


def pickleStore( savethings , filename ):
    dbfile = open( filename , 'wb' )
    pickle.dump( savethings , dbfile )
    dbfile.close()
    return


def pickleOpen( filename ):
    file_to_read = open( filename , "rb" )
    p = pickle.load( file_to_read )
    return p


def softmax( vector ):
	e = np.exp( vector )
	return e / e.sum()


def MAP( rel , ans , q ):
    t = 0
    for i in range( 0 , q ):
        s , sss = 0 , {}
        # 這行是 revelant set
        a = rel.split()
        # 這行是 answer set
        b = set( ans.split() )
        for index , doc in enumerate( a ):
            if doc in b:
                s += 1
                sss [ s ] = s / ( index + 1 )
        t += sum( sss.values() ) / len( b )
    return round( ( t / q ) * 10000.0  ) / 10000


def Train_alpha( start , stop , interval ):
    
    que_top_dict = pickleOpen( dic_save + "train_que_top_dict.pkl" )
    bestalpha = 0
    bestamap  = 0
    for alpha in np.arange( start + interval , stop + interval , interval ):
        new  = {}
        for query_name , d_v in que_top_dict.items():
            if query_name in d_list:
                if query_name not in new.keys():
                    new[ query_name ] = {}
                for doc_name , softmax_score in d_v.items():
                    new[ query_name ][ doc_name ] = np.log2( save[ query_name ][ doc_name ] ) + alpha * np.log2( softmax_score )
        pickleStore( new , dic_rerank + "train_ques_docs_alpha_reranking-{}.pkl".format( alpha ) )

        map_list = []
        for query_name , d_v in new.items():
            row = [ str( query_name ) , que_pos_dict[ query_name ] , " ".join( sorted( d_v , key=d_v.get , reverse=True ) ) ]
            map = MAP( row[2] , row[1] , len( row[2].split() ) )
            print( "{0} : {1}".format( query_name , map ) )
        aMAP = sum( map_list ) / len( new )
        print( "Total aMAP : {0}, for alpha {1}".format( aMAP , alpha ) )
        if aMAP > bestamap:
            bestamap = aMAP
            bestalpha = alpha
    return bestamap , bestalpha


if __name__ == "__main__":

    dic_sources = 'ntust-ir2020-homework6/'
    dic_save    = dic_sources + 'save/'
    dic_rerank  = dic_save + 'rerank/'

    ## reproduce alpha ranking
    d_list = pickleOpen( dic_save + "alpha_querys_docs_list.pkl" )
    que_pos_dict = pickleOpen( dic_save + "train_que_pos_dict.pkl" )
    trainer_alpha_result = pickleOpen( dic_save + "trainer_alpha_result.pkl" )
    save = {}
    tmp  = {}
    pre_result = trainer_alpha_result[0]
    with open( dic_save + 'train_for_alpha_train.csv' , newline='' ) as csvfile:
        spamreader = csv.reader( csvfile , delimiter=',' )
        next(spamreader)
        for c , row in enumerate( spamreader ):
            if row[0] not in save.keys():
                save[ row[0] ] = {}
            for tup in zip( row[2].split() , pre_result[c].tolist() ):
                save[ row[0] ][ tup[0] ] = tup[1]
        # sort
        save = { query_name : dict( sorted( d_v.items(), key=lambda item: item[1] , reverse=True ) ) for query_name , d_v in save.items() }
        # compute softmax
        for query_name , d_v in save.items():
            tmp[ query_name ] = softmax( np.array( list( save[ query_name ].values() ) , dtype=np.float64 ) ).tolist()
        save = { query_name : dict( zip( list( save[ query_name ].keys() ) , tmp[ query_name ] ) ) for query_name in save.keys() }

    bestamap , bestalpha = Train_alpha( 0 , 5 , 0.01 )
    print( "Best aMAP: {0}, Best Alpha: {1}".format( bestamap , bestalpha ) )

    # recalculate new rereanking score again
    new  = {}
    save = {}
    with open( dic_save + 'test.csv' , newline='' ) as csvfile:
        spamreader = csv.reader( csvfile , delimiter=',' )
        next(spamreader)
        for c , row in enumerate( spamreader ):
            if row[0] not in save.keys():
                save[ row[0] ] = {}
            for tup in zip( row[2].split() , pre_result[c].tolist() ):
                save[ row[0] ][ tup[0] ] = tup[1]
        # sort
        save = { query_name : dict( sorted( d_v.items(), key=lambda item: item[1] , reverse=True ) ) for query_name , d_v in save.items() }
        # compute softmax
        for query_name , d_v in save.items():
            tmp[ query_name ] = softmax( np.array( list( save[ query_name ].values() ) , dtype=np.float64 ) ).tolist()
        save = { query_name : dict( zip( list( save[ query_name ].keys() ) , tmp[ query_name ] ) ) for query_name in save.keys() }

    que_top_dict = pickleOpen( dic_save + "train_que_top_dict.pkl" )
    for query_name , d_v in que_top_dict.items():
        if query_name not in new.keys():
            new[ query_name ] = {}
        for doc_name , softmax_score in d_v.items():
            new[ query_name ][ doc_name ] = np.log2( save[ query_name ][ doc_name ] ) + bestalpha * np.log2( softmax_score )
    pickleStore( new , dic_save + "train_ques_docs_best_rerank.pkl" )

    # save result
    with open( dic_save + "test_ques_docs_best_rerank.csv" , "a" ) as writefile:
        writefile.write( "query_id,ranked_doc_ids\n" )
        for query_name , d_v in new.items():
            append = ",".join( [ str( query_name ) , " ".join( sorted( d_v , key=d_v.get , reverse=True ) ) ] )
            writefile.write( append + "\n" )
