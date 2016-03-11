SET search_path TO mimiciii;

-- Size [37166]
DROP TABLE rbc_fusion_ce;
CREATE TABLE rbc_fusion_ce AS (
    SELECT 
        t.subject_id,
        t.icustay_id,
        t.charttime,
        t.value,
        t.stopped
    FROM (
        SELECT 
            p.subject_id,
            p.icustay_id,
            ce.charttime,
            ce.itemid,
            ce.value,
            ce.stopped
        FROM pop p LEFT JOIN chartevents ce 
          ON p.icustay_id = ce.icustay_id
    ) t
    WHERE t.itemid IN (
        833,
        3799,
        4197
    ) AND t.value IS NOT NULL AND t.value <> ''
);

-- Size [11601]
DROP TABLE rbc_fusion_cv;
CREATE TABLE rbc_fusion_cv AS (
    SELECT 
        t.subject_id,
        t.icustay_id,
        t.charttime,
        t.value,
        t.stopped
    FROM (
        SELECT 
            p.subject_id,
            p.icustay_id,
            cv.itemid,
            cv.charttime,
            cv.amount as value,
            cv.stopped
        FROM pop p LEFT JOIN inputevents_cv cv
          ON p.icustay_id = cv.icustay_id
    ) t
    WHERE t.itemid IN (
        30001,
        30004,
        30104,
        30106,
        30179
    ) AND t.value IS NOT NULL
);
ALTER TABLE rbc_fusion_cv ALTER COLUMN value TYPE varchar(255);
ALTER TABLE rbc_fusion_cv ALTER COLUMN stopped TYPE varchar(50);


-- Size [7509]
DROP TABLE rbc_fusion_mv;
CREATE TABLE rbc_fusion_mv AS (
    SELECT 
        t.subject_id,
        t.icustay_id,
        t.charttime,
        t.value,
        t.stopped
    FROM (
        SELECT 
            p.subject_id,
            p.icustay_id,
            mv.itemid,
            mv.endtime AS charttime,
            mv.amount AS value,
            'Stopped' AS stopped
        FROM pop p LEFT JOIN inputevents_mv mv
          ON p.icustay_id = mv.icustay_id
    ) t
    WHERE t.itemid IN (
        225168,
        226368,
        226370,
        227070
    ) AND t.value IS NOT NULL
);
ALTER TABLE rbc_fusion_mv ALTER COLUMN value TYPE varchar(255);
ALTER TABLE rbc_fusion_mv ALTER COLUMN stopped TYPE varchar(50);

-- Total Size [56276]
DROP TABLE rbc_fusion;
CREATE TABLE rbc_fusion 
AS (
    SELECT ce.* 
    FROM rbc_fusion_ce ce
    UNION ALL
    SELECT cv.*
    FROM rbc_fusion_cv cv
    UNION ALL 
    SELECT mv.*
    FROM rbc_fusion_mv mv
) ORDER BY subject_id ASC;
       
DROP TABLE rbc_fusion_ce;
DROP TABLE rbc_fusion_cv;
DROP TABLE rbc_fusion_mv;

