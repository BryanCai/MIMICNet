SET search_path TO mimiciii;

DROP TABLE public.systolic;
CREATE TABLE public.systolic 
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
  WHERE var.itemid IN (51, 220050)
);
\COPY public.systolic TO '../Data/systolic.csv' DELIMITER ',' CSV HEADER;