
-- Criterions 1: first time in ICU, Adult patient, between 12 - 96 hours of ICU stay
-- Criterion 2: 1) Exclude CMO, 2) Exclude DNR/DNI, 3) Include only Full Code, 4) No NSICU, CSICU 

SET search_path TO mimiciii;

-- Initialize POP
-- Size [27876]
-- This was changed to hospstay_seq
DROP TABLE prepop;
CREATE TABLE prepop
AS (
  WITH summary AS (
    SELECT 
      icud.*,
      ROW_NUMBER() OVER (PARTITION BY icud.subject_id 
                         ORDER BY icud.admittime ASC) AS rk
    FROM icustay_detail icud
    WHERE icud.hospstay_seq = 1 AND icud.icustay_seq = 1 AND icud.age >= 15 AND icud.los_icu >= 0.5 AND icud.los_icu <= 4
  )

  SELECT 
    su.subject_id,
    su.hadm_id,
    su.icustay_id,
    su.gender,
    su.dob,
    su.dod,
    su.ethnicity,
    su.admission_type,
    su.admittime,
    su.dischtime,
    su.hospital_expire_flag,
    su.hospstay_seq,
    su.first_hosp_stay,
    su.intime,
    su.outtime,
    su.age,
    su.los_icu,
    su.icustay_seq
  FROM summary su
  WHERE su.rk = 1
);

-- Size [43073438]
DROP TABLE merged;
CREATE TABLE merged
AS (

  -- Merge the patient data with chartevents

  SELECT 
    pp.subject_id,
    pp.hadm_id,
    pp.icustay_id,
    pp.gender,
    pp.dob,
    pp.dod,
    pp.ethnicity,
    pp.admission_type,
    pp.admittime,
    pp.dischtime,
    pp.hospital_expire_flag,
    pp.hospstay_seq,
    pp.first_hosp_stay,
    pp.intime,
    pp.outtime,
    pp.age,
    pp.los_icu,
    pp.icustay_seq,
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
  FROM prepop pp
  LEFT JOIN chartevents ce ON pp.icustay_id = ce.icustay_id
);

-- Find PACEMAKER data
-- size [3161]

DROP TABLE pacemaker;
CREATE TABLE pacemaker
AS (
  WITH summary AS (
    SELECT 
      m.icustay_id, 
      m.value,
      ROW_NUMBER() OVER (PARTITION BY m.icustay_id 
                          ORDER BY m.icustay_id DESC) AS rk
    FROM merged m
    WHERE m.itemid IN (515, 223956, 224844)
  )

  -- Select the first from each row.

  SELECT 
    s.icustay_id,
    s.value
  FROM summary s
  WHERE s.rk = 1
);

-- Find RISK FOR FALLS data
-- Size [9813]

DROP TABLE riskfalls;
CREATE TABLE riskfalls 
AS (
  WITH summary AS (
    SELECT 
      m.icustay_id, 
      m.value,
      ROW_NUMBER() OVER (PARTITION BY m.icustay_id 
                          ORDER BY m.icustay_id DESC) AS rk
    FROM merged m
    WHERE m.itemid IN (1484, 223754)
  )

  -- Select the first from each row.

  SELECT 
    s.icustay_id,
    s.value
  FROM summary s
  WHERE s.rk = 1
);

-- Find all the unique icustay_id's in merged table
-- Size [27876]
DROP TABLE allids;
CREATE TABLE allids
AS (
  SELECT DISTINCT m.icustay_id
  FROM merged m
);

-- Find all the bad id's we are trying to avoid
-- Size [1824]
DROP TABLE badids;
CREATE TABLE badids
AS (

  -- Grab out only the events WITHOUT full code for care protocol

  WITH tmp AS (
    SELECT *
    FROM merged m
    WHERE m.itemid = 128 AND m.value != 'Full Code'
  )

  SELECT DISTINCT tmp.icustay_id
  FROM tmp 
);

-- Find the remaining ids that are good.
-- Size [26052]
DROP TABLE goodids;
CREATE TABLE goodids
AS (
  SELECT al.icustay_id
  FROM allids al
  WHERE al.icustay_id NOT IN (
    SELECT bad.icustay_id
    FROM badids bad
  )
);

-- Size [26052]
DROP TABLE pop;
CREATE TABLE pop
AS (
  WITH tmp AS (
    SELECT foo.*
    FROM (
      SELECT 
        pp.*,
        info.first_careunit
      FROM prepop pp
      LEFT JOIN icustays info ON pp.icustay_id = info.icustay_id
    ) foo
    WHERE foo.first_careunit NOT IN ('NICU') AND 
      foo.icustay_id IN (
        SELECT good.icustay_id
        FROM goodids good
      )
  )

  SELECT last.*
  FROM (
    SELECT 
      tmp2.*,
      ri.value AS riskfalls
    FROM (
      SELECT 
        tmp.*,
        pa.value AS pacemaker 
      FROM tmp LEFT JOIN pacemaker pa ON tmp.icustay_id = pa.icustay_id
    ) tmp2 LEFT JOIN riskfalls ri ON tmp2.icustay_id = ri.icustay_id
  ) last
);

DROP TABLE prepop;
DROP TABLE merged;
DROP TABLE pacemaker;
DROP TABLE riskfalls;
DROP TABLE allids;
DROP TABLE badids;
DROP TABLE goodids;
