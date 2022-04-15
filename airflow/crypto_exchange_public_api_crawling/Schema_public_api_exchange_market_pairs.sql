/* =============================================================
     STEP01 : truncate Data Lake table before COPY COMMAND
   ============================================================= */

create table if not exists "Data Lake"
( base_currency           varchar(50)
, etime_at                varchar(255)
, open_price              numeric(38,18)
, high_price              numeric(38,18)
, low_price               numeric(38,18)
, krw_volume              numeric(38,18)
, base_volume             numeric(38,18)
, exchange                varchar(50))
;

truncate table "Data Lake";


create table if not exists "Data Warehouse"
( basis_dy                varchar(50)
, base_currency           varchar(50)
, etime_at                varchar(255)
, open_price              numeric(38,18)
, high_price              numeric(38,18)
, low_price               numeric(38,18)
, krw_volume              numeric(38,18)
, base_volume             numeric(38,18)
, exchange                varchar(50)
, dw_load_dt              timestamp without time zone not null
, constraint PK_NAME primary key (basis_dy, etime_at))
;

/* =============================================================
     STEP02 : COPY COMMAND from Data Lake -> Data Warehouse
   ============================================================= */

copy "Data Lake"
from 'Security'
-- IAM_ROLE 'Security' --dev
IAM_ROLE 'Security' --prod
region as 'Security'
statupdate off compupdate off
dateformat 'Security'
emptyasnull
JSON as 'Security / Jsonpaths'
acceptinvchars
gzip
;

/* =============================================================
     STEP03 : Dedup (Make sure no duplicated rows)
   ============================================================= */

--delete from "Data Warehouse"
--  where basis_dy = '{{ tomorrow_ds_nodash }}';

 delete from "Data Warehouse"
 using "Data Lake"
       /** delete with primary values **/
       where "Data Warehouse".etime_at             = "Data Lake".etime_at

;

/* =============================================================
     STEP04 : Insert from Data Lake to Data Warehouse
   ============================================================= */

insert into "Data Warehouse"
(  basis_dy
 , base_currency
 , etime_at
 , open_price
 , high_price
 , low_price
 , krw_volume
 , base_volume
 , exchange
 , dw_load_dt
)
select '{{ tomorrow_ds_nodash }}'  as basis_dy
 , base_currency
 , etime_at
 , open_price
 , high_price
 , low_price
 , krw_volume
 , base_volume
 , exchange
 , current_timestamp             as dw_load_dt
from "Data Lake"
;
