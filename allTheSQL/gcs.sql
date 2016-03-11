-- retrieves the glasgow coma score for each adult patient 
-- only works for patients recorded with carevue 

SET search_path TO mimiciii;
DROP TABLE gcs;
CREATE TABLE gcs
AS

with agetbl as
(
	select ad.subject_id, ad.hadm_id
       from mimiciii.admissions ad
       inner join mimiciii.patients p
       on ad.subject_id = p.subject_id 
       where
       -- filter to only adults
        ( 
		(extract(DAY from ad.admittime - p.dob) 
			+ extract(HOUR from ad.admittime - p.dob) /24
			+ extract(MINUTE from ad.admittime - p.dob) / 24 / 60
			) / 365.25 
	) > 15
)
select bucket, count(*) from (
  select width_bucket(valuenum, 1, 30, 30) as bucket
    from mimiciii.chartevents ce
    inner join agetbl    
    on ce.subject_id = agetbl.subject_id
   where itemid in (198)
       )as gcs
       group by bucket order by bucket;
