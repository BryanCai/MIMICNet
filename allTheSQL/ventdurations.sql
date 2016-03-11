-- This query extracts the duration of mechanical ventilation
set search_path to mimiciii;

drop table if exists ventdurations;
create table ventdurations as
-- First create a staging table with only vent settings from chart events
with ventsettings as
(
  select
    subject_id, icustay_id, charttime
    -- case statement determining whether it is an instance of mech vent
    , max(
      case
        when itemid is null or value is null then 0 -- can't have null values
        when itemid = 720 and value != 'Other/Remarks' THEN 1  -- VentTypeRecorded
        when itemid = 467 and value = 'Ventilator' THEN 1 -- O2 delivery device == ventilator
        when itemid = 648 and value = 'Intubated/trach' THEN 1 -- Speech = intubated
        when itemid in
          (
          445, 448, 449, 450, 1340, 1486, 1600, 224687 -- minute volume
          , 639, 654, 681, 682, 683, 684,224685,224684,224686 -- tidal volume
          , 218,436,535,444,459,224697,224695,224696,224746,224747 -- High/Low/Peak/Mean/Neg insp force ("RespPressure")
          , 221,1,1211,1655,2000,226873,224738,224419,224750,227187 -- Insp pressure
          , 543 -- PlateauPressure
          , 5865,5866,224707,224709,224705,224706 -- APRV pressure
          , 60,437,505,506,686,220339,224700 -- PEEP
          , 3459 -- high pressure relief
          , 501,502,503,224702 -- PCV
          , 223,667,668,669,670,671,672 -- TCPCV
          , 157,158,1852,3398,3399,3400,3401,3402,3403,3404,8382,227809,227810 -- ETT
          , 224701 -- PSVlevel
          )
          THEN 1
        else 0
      end
      ) as MechVent
      , max(
        case when itemid is null or value is null then 0
          when itemid = 640 and value = 'Extubated' then 1
          when itemid = 640 and value = 'Self Extubation' then 1
        else 0
        end
        )
        as Extubated
      , max(
        case when itemid is null or value is null then 0
          when itemid = 640 and value = 'Self Extubation' then 1
        else 0
        end
        )
        as SelfExtubated

  from mimiciii.chartevents ce
  where value is not null
  and itemid in
  (
      640 -- extubated
      , 648 -- speech
      , 720 -- vent type
      , 467 -- O2 delivery device
      , 445, 448, 449, 450, 1340, 1486, 1600, 224687 -- minute volume
      , 639, 654, 681, 682, 683, 684,224685,224684,224686 -- tidal volume
      , 218,436,535,444,459,224697,224695,224696,224746,224747 -- High/Low/Peak/Mean/Neg insp force ("RespPressure")
      , 221,1,1211,1655,2000,226873,224738,224419,224750,227187 -- Insp pressure
      , 543 -- PlateauPressure
      , 5865,5866,224707,224709,224705,224706 -- APRV pressure
      , 60,437,505,506,686,220339,224700 -- PEEP
      , 3459 -- high pressure relief
      , 501,502,503,224702 -- PCV
      , 223,667,668,669,670,671,672 -- TCPCV
      , 157,158,1852,3398,3399,3400,3401,3402,3403,3404,8382,227809,227810 -- ETT
      , 224701 -- PSVlevel
  )
  group by subject_id, icustay_id, charttime
)
-- now we convert CHARTTIME of ventilator settings into durations
, vd1 as
(
select
    subject_id
    , icustay_id
    -- this carries over the previous charttime which had a mechanical ventilation event
    , case
        when MechVent=1 then
          LAG(CHARTTIME, 1) OVER (partition by icustay_id, MechVent order by charttime)
        else null
      end as charttime_lag
    , charttime
    , MechVent
    , Extubated
    , SelfExtubated

    -- if this is a mechanical ventilation event, we calculate the time since the last event
    , case
        -- if the current observation indicates mechanical ventilation is present
        when MechVent=1 then
        -- copy over the previous charttime where mechanical ventilation was present
          CHARTTIME - (LAG(CHARTTIME, 1) OVER (partition by icustay_id, MechVent order by charttime))
        else null
      end as ventduration

    -- now we determine if the current mech vent event is a "new", i.e. they've just been intubated
    , case
      -- if there is an extubation flag, we mark any subsequent ventilation as a new ventilation event
        when Extubated = 1 then 0 -- extubation is *not* a new ventilation event, the *subsequent* row is
        when
          LAG(Extubated,1)
          OVER
          (
          partition by icustay_id, case when MechVent=1 or Extubated=1 then 1 else 0 end
          order by charttime
          )
          = 1 then 1
          -- if there is less than 8 hours between vent settings, we do not treat this as a new ventilation event
        when (CHARTTIME - (LAG(CHARTTIME, 1) OVER (partition by icustay_id, MechVent order by charttime))) <= interval '8' hour
          then 0
      else 1
      end as newvent
FROM
  ventsettings
)
, vd2 as
(
select vd1.*
-- create a cumulative sum of the instances of new ventilation
-- this results in a monotonic integer assigned to each instance of ventilation
, case when MechVent=1 or Extubated = 1 then
    SUM( newvent )
    OVER ( partition by icustay_id order by charttime )
  else null end
  as ventnum
from vd1
-- now we can isolate to just rows with ventilation settings/extubation settings
-- (before we had rows with extubation flags)
-- this removes any null values for newvent
where
  MechVent = 1 or Extubated = 1
)
,
-- startTimes
ventStarts as
( select subject_id, icustay_id, min(charttime) as charttime, ventnum as value, 0 as stopped
from vd2
group by subject_id, icustay_id, ventnum
order by subject_id, icustay_id
)
,
ventEnds as 
(select subject_id, icustay_id, max(charttime) as charttime, ventnum as value, 1 as stopped
from vd2
group by subject_id, icustay_id, ventnum
order by subject_id, icustay_id
)
select * from ventStarts
union all 
select * from ventEnds;