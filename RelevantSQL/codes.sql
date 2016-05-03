SET search_path TO mimiciii;

DROP TABLE public.codes;
CREATE TABLE public.codes 
AS (
  SELECT 
    var.subject_id,
    var.icustay_id,
    var.icd9_code
  FROM (
    SELECT
      p.*,
      dia.row_id,
      dia.seq_num,
      dia.icd9_code
    FROM pop p 
      LEFT JOIN diagnoses_icd dia ON p.hadm_id = dia.hadm_id
  ) var
  WHERE var.seq_num = 1
  AND var.subject_id < 30000
);
\COPY public.codes TO '../Data/codes.csv' DELIMITER ',' CSV HEADER;
