SET search_path TO mimiciii;

-- Total Size [108139]

-- Size [13]
DROP TABLE vaso_ce;
CREATE TABLE vaso_ce AS (
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
        5747,
        5329,
        4501,
        5805,
        3112,
        5843,
        5682,
        1136,
        2445,
        1222,
        6255,
        2334,
        7341,
        2561,
        1327,
        2248,
        2765
    ) AND t.value IS NOT NULL AND t.value <> ''
);
UPDATE vaso_ce SET value = 0 WHERE value = 'OFF';
UPDATE vaso_ce SET value = 0 WHERE value = 'off';

-- Size [92331]
DROP TABLE vaso_cv;
CREATE TABLE vaso_cv AS (
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
        30042,
        30306,
        30043,
        30307,
        30044,
        30309,
        30119,
        30047,
        30120,
        30125,
        30127,
        30128,
        30051,
        42802,
        42273
    ) AND t.value IS NOT NULL
);
ALTER TABLE vaso_cv ALTER COLUMN value TYPE varchar(255);
ALTER TABLE vaso_cv ALTER COLUMN stopped TYPE varchar(50);

-- Size [15795]
DROP TABLE vaso_mv;
CREATE TABLE vaso_mv AS (
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
        211653,
        221662,
        221289,
        221986,
        221906,
        222315
    ) AND t.value IS NOT NULL
);
ALTER TABLE vaso_mv ALTER COLUMN value TYPE varchar(255);
ALTER TABLE vaso_mv ALTER COLUMN stopped TYPE varchar(50);

DROP TABLE vaso;
CREATE TABLE vaso 
AS (
    SELECT ce.* 
    FROM vaso_ce ce
    UNION ALL
    SELECT cv.*
    FROM vaso_cv cv
    UNION ALL 
    SELECT mv.*
    FROM vaso_mv mv
) ORDER BY subject_id ASC;
       
DROP TABLE vaso_ce;
DROP TABLE vaso_cv;
DROP TABLE vaso_mv;

