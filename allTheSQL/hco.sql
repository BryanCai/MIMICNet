-- retrieves all recorded bicarbonate levels 

SET search_path TO mimiciii;
DROP TABLE hco;
CREATE TABLE hco 
AS 

select bucket, count(*) from (
       select width_bucket(valuenum, 0, 231, 231) as bucket 
       from mimiciii.labevents 
       where itemid in (50803, 50804, 50882)
       ) as hco
       group by bucket order by bucket; 

