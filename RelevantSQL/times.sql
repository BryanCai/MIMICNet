SET search_path TO mimiciii;

DROP TABLE public.times;
CREATE TABLE public.times 
AS (
  SELECT 
    p.subject_id,
    p.icustay_id,
    p.intime,
    p.outtime
  FROM pop p
  WHERE var.subject_id < 30000
);
\COPY public.times TO '../Data/times.csv' DELIMITER ',' CSV HEADER;
