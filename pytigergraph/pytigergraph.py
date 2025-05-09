import pyTigerGraph as tg
import pandas as pd
conn=tg.TigerGraphConnection(host="https://f1e202f59b9d461d96a7c56157ca2e7e.i.tgcloud.io",graphname="SampleGraph")
#conn=tg.TigerGraphConnection(host="https://f1e202f59b9d461d96a7c56157ca2e7e.i.tgcloud.io",username="User1",password="User1", graphname="SampleGraph")
conn.apiToken=conn.getToken("lkq5u1717b5ape181ti47o4pv9rpc1eg")[0]
print (conn)
# print(conn.gsql(
# '''
# USE GRAPH SampleGraph
# CREATE SCHEMA_CHANGE job job1 FOR GRAPH SampleGraph{
# ADD VERTEX Address(PRIMARY_ID uniqueid STRING,street STRING,Country STRING)WITH primary_id_as_attribute="TRUE";
# ADD VERTEX Person(PRIMARY_ID uniqueid STRING,name STRING,age INT)WITH primary_id_as_attribute="TRUE";
# ADD UNDIRECTED Edge Residence(from Person, to Address, Date_time datetime );
# }
# RUN SCHEMA_CHANGE JOB job1
# DROP JOB job1
 
# '''
# ))


person_csv=pd.read_csv("C:/Users/G_Subramanian/Downloads/person.csv")
print(person_csv)
person_df=pd.DataFrame()
person_df=person_csv
person_df['uniqueid'] = person_df['uniqueid'].astype(str)
personvt=conn.upsertVertexDataFrame(person_df,"Person","uniqueid",attributes={"uniqueid":"uniqueid","name":"name","age":"age"}) #attributes={"vertexattribute":"dataframe columnname"}
#upsertVertexDataFrame(dataframe, vertexname, primaryid column name from dataframe, attribute mapping)
print(personvt)

edge_csv=pd.read_csv("edge.csv")
ed=pd.DataFrame(edge_csv)
ed['id']=ed['id'].astype(str)
ed['phonenum']=ed['phonenum'].astype(str)
ed1=conn.upsertEdgeDataFrame(ed,"Person","person_phone","Phone",from_id="id",to_id="phonenum",attributes={})
#upsertEdgeDataFrame(dataframe,start vertex name, edge name,target vertex name, from primary id from dataframe , to primary id from dataframe )