SET search_path TO mimiciii;

DROP TABLE public.FiO2;
CREATE TABLE public.FiO2 
AS (
  SELECT 
    var.subject_id,
    var.icustay_id,
    var.charttime - var.intime,
    var.valuenum AS value
  FROM (
    SELECT
      p.*,
      ce.row_id,
      ce.itemid,
      ce.charttime,
      ce.storetime,
      ce.cgid,
      ce.value,
      ce.valuenum,
      ce.valueuom,
      ce.warning,
      ce.error,
      ce.resultstatus,
      ce.stopped
    FROM pop p 
      LEFT JOIN chartevents ce ON p.icustay_id = ce.icustay_id
  ) var
  WHERE var.itemid IN (190, 2981, 3420, 223835)
  AND var.subject_id < 30000
  AND var.charttime BETWEEN var.intime AND var.outtime

);
\COPY public.FiO2 TO '../Data/FiO2.csv' DELIMITER ',' CSV HEADER;