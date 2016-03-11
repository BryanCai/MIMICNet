SET search_path TO mimiciii;

-- Size [708726]
DROP TABLE spontaneousrr;
CREATE TABLE spontaneousrr 
AS (
  SELECT 
    var.subject_id,
    var.icustay_id,
    var.charttime,
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
  WHERE var.itemid IN (614, 615, 618, 224689) AND
    var.valuenum >= 2 AND 
    var.valuenum <= 80
);
