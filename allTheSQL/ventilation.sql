-- Only take the data from ventdurations that exists in pop;

SET search_path TO mimiciii;

DROP TABLE ventilation;
CREATE TABLE ventilation 
AS (
  SELECT v.*
  FROM pop p LEFT JOIN ventdurations v
    ON p.icustay_id = v.icustay_id
) ORDER BY subject_id ASC;
