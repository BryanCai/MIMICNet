-- Pick out the mortality times. 
-- Size [26052]

SET search_path TO mimiciii;

DROP TABLE mortality;
CREATE TABLE mortality 
AS (
  SELECT 
    p.subject_id,
    p.icustay_id,
    adm.deathtime AS charttime, 
    adm.hospital_expire_flag AS value,
    0 AS stopped
  FROM
    pop p LEFT JOIN admissions adm ON p.hadm_id=adm.hadm_id
  WHERE lower(diagnosis) NOT LIKE '%organ donor%'
);

