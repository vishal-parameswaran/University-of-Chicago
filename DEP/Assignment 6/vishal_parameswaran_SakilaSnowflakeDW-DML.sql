/***********************************************
**                MSc ANALYTICS 
**     DATA ENGINEERING PLATFORMS (MSCA 31012)
** File:   Sakila Snowflake DDL - Assignment 5
** Desc:   ETL/DML for the Sakila Snowflake Dimensional model
** Auth:   Shreenidhi Bharadwaj, Ashish Pujari, Audrey Salerno
** Date:   04/08/2018, Last updated 02/09/2021
************************************************/

-- ---------------------------------------
USE SAKILA_SNOWFLAKE;

-- -----------------------------------------------------
-- Populate Time dimension
-- -----------------------------------------------------

INSERT INTO NUMBERS_SMALL
VALUES (0),
       (1),
       (2),
       (3),
       (4),
       (5),
       (6),
       (7),
       (8),
       (9);

INSERT INTO NUMBERS
SELECT THOUSANDS.NUMBER * 1000 + HUNDREDS.NUMBER * 100 + TENS.NUMBER * 10 + ONES.NUMBER
FROM NUMBERS_SMALL THOUSANDS,
     NUMBERS_SMALL HUNDREDS,
     NUMBERS_SMALL TENS,
     NUMBERS_SMALL ONES
LIMIT 1000000;

INSERT INTO DIM_DATE (DATE_KEY, DATE)
SELECT NUMBER,
       DATE_ADD('2005-01-01',
                INTERVAL NUMBER DAY)
FROM NUMBERS
WHERE DATE_ADD('2005-01-01',
               INTERVAL NUMBER DAY) BETWEEN '2005-01-01' AND '2017-01-01'
ORDER BY NUMBER;

SET SQL_SAFE_UPDATES = 0;
UPDATE DIM_DATE
SET TIMESTAMP   = UNIX_TIMESTAMP(DATE),
    DAY_OF_WEEK = DATE_FORMAT(DATE, '%W'),
    WEEKEND     = IF(DATE_FORMAT(DATE, '%W') IN ('Saturday', 'Sunday'),
                     'Weekend',
                     'Weekday'),
    MONTH       = DATE_FORMAT(DATE, '%M'),
    YEAR        = DATE_FORMAT(DATE, '%Y'),
    MONTH_DAY   = DATE_FORMAT(DATE, '%d');

UPDATE DIM_DATE
SET WEEK_STARTING_MONDAY = DATE_FORMAT(DATE, '%v');

-- -----------------------------------------------------
-- Copy Data from sakila database 
-- -----------------------------------------------------
# dim_actor table 
# insert data into the dim_actor table from sakila models actor table 
INSERT INTO SAKILA_SNOWFLAKE.DIM_ACTOR (ACTOR_ID,
                                        ACTOR_FIRST_NAME,
                                        ACTOR_LAST_NAME,
                                        ACTOR_LAST_UPDATE)
    (SELECT ACTOR_ID,
            FIRST_NAME,
            LAST_NAME,
            LAST_UPDATE
     FROM SAKILA.ACTOR);

# dim_staff table
# insert data into the dim_staff table from sakila models staff table 
INSERT INTO SAKILA_SNOWFLAKE.DIM_STAFF (STAFF_ID,
                                        STAFF_FIRST_NAME,
                                        STAFF_LAST_NAME,
                                        STAFF_STORE_ID,
                                        STAFF_LAST_UPDATE)
    (SELECT STAFF_ID,
            FIRST_NAME,
            LAST_NAME,
            STORE_ID,
            LAST_UPDATE
     FROM SAKILA.STAFF);


# dim_location_country
# insert data into the dim_location_country table from sakila models country table 
INSERT INTO SAKILA_SNOWFLAKE.DIM_LOCATION_COUNTRY (LOCATION_COUNTRY_ID,
                                                   LOCATION_COUNTRY_NAME,
                                                   LOCATION_COUNTRY_LAST_UPDATE)
    (SELECT COUNTRY_ID,
            COUNTRY,
            LAST_UPDATE
     FROM SAKILA.COUNTRY);

# dim_location_city
# insert data into the dim_location_city table from sakila model city table 
INSERT INTO SAKILA_SNOWFLAKE.DIM_LOCATION_CITY(LOCATION_COUNTRY_KEY,
                                               LOCATION_CITY_ID,
                                               LOCATION_CITY_NAME,
                                               LOCATION_CITY_LAST_UPDATE)
    (SELECT LOCATION_COUNTRY_KEY,
            CITY_ID,
            CITY,
            S_CITY.LAST_UPDATE
     FROM SAKILA.COUNTRY AS S_COUNTRY,
          SAKILA.CITY AS S_CITY,
          SAKILA_SNOWFLAKE.DIM_LOCATION_COUNTRY AS DIM_COUNTRY
     WHERE S_CITY.COUNTRY_ID = S_COUNTRY.COUNTRY_ID
       AND S_CITY.COUNTRY_ID = DIM_COUNTRY.LOCATION_COUNTRY_ID);


# dim_location_address
# insert data into the dim_location_address table from sakila model city table 
INSERT INTO SAKILA_SNOWFLAKE.DIM_LOCATION_ADDRESS(LOCATION_CITY_KEY,
                                                  LOCATION_ADDRESS_ID,
                                                  LOCATION_ADDRESS,
                                                  LOCATION_ADDRESS_POSTAL_CODE,
                                                  LOCATION_ADDRESS_LAST_UPDATE)
    (SELECT DIM_CITY.LOCATION_CITY_KEY,
            ADDRESS_ID,
            ADDRESS,
            POSTAL_CODE,
            LAST_UPDATE
     FROM SAKILA.ADDRESS AS S_ADDRESS,
          SAKILA_SNOWFLAKE.DIM_LOCATION_CITY AS DIM_CITY
     WHERE S_ADDRESS.CITY_ID = DIM_CITY.LOCATION_CITY_ID);


# dim_store table
# insert data into the dim_store table from sakila models staff table 
INSERT INTO SAKILA_SNOWFLAKE.DIM_STORE (LOCATION_ADDRESS_KEY,
                                        STORE_LAST_UPDATE,
                                        STORE_ID,
                                        STORE_MANAGER_STAFF_ID,
                                        STORE_MANAGER_FIRST_NAME,
                                        STORE_MANAGER_LAST_NAME)
    (SELECT S_ADDR.LOCATION_ADDRESS_KEY,
            S_STORE.LAST_UPDATE,
            S_STORE.STORE_ID,
            MANAGER_STAFF_ID,
            FIRST_NAME,
            LAST_NAME
     FROM SAKILA.STORE AS S_STORE,
          DIM_LOCATION_ADDRESS AS S_ADDR,
          SAKILA.STAFF AS S_STAFF
     WHERE S_STORE.ADDRESS_ID = S_ADDR.LOCATION_ADDRESS_ID
       AND S_STAFF.STAFF_ID = S_STORE.MANAGER_STAFF_ID);


# dim_customer
INSERT INTO SAKILA_SNOWFLAKE.DIM_CUSTOMER (LOCATION_ADDRESS_KEY,
                                           CUSTOMER_LAST_UPDATE,
                                           CUSTOMER_ID,
                                           CUSTOMER_FIRST_NAME,
                                           CUSTOMER_LAST_NAME,
                                           CUSTOMER_EMAIL,
                                           CUSTOMER_ACTIVE,
                                           CUSTOMER_CREATED)
    (SELECT LOCATION_ADDRESS_KEY,
            LAST_UPDATE,
            CUSTOMER_ID,
            FIRST_NAME,
            LAST_NAME,
            EMAIL,
            ACTIVE,
            CREATE_DATE
     FROM SAKILA.CUSTOMER AS S_CUST,
          DIM_LOCATION_ADDRESS AS S_ADDR
     WHERE S_CUST.ADDRESS_ID = S_ADDR.LOCATION_ADDRESS_ID);



# dim_film
# insert data into the dim_store table from sakila models staff table 
INSERT INTO SAKILA_SNOWFLAKE.DIM_FILM (FILM_ID,
                                       FILM_LAST_UPDATE,
                                       FILM_TITLE,
                                       FILM_DESCRIPTION,
                                       FILM_RELEASE_YEAR,
                                       FILM_LANGUAGE,
                                       FILM_RENTAL_DURATION,
                                       FILM_RENTAL_RATE,
                                       FILM_DURATION,
                                       FILM_REPLACEMENT_COST,
                                       FILM_RATING_CODE,
                                       FILM_RATING_TEXT,
                                       FILM_HAS_TRAILERS,
                                       FILM_HAS_COMMENTARIES,
                                       FILM_HAS_DELETED_SCENES,
                                       FILM_HAS_BEHIND_THE_SCENES,
                                       FILM_IN_CATEGORY_ACTION,
                                       FILM_IN_CATEGORY_ANIMATION,
                                       FILM_IN_CATEGORY_CHILDREN,
                                       FILM_IN_CATEGORY_CLASSICS,
                                       FILM_IN_CATEGORY_COMEDY,
                                       FILM_IN_CATEGORY_DOCUMENTARY,
                                       FILM_IN_CATEGORY_DRAMA,
                                       FILM_IN_CATEGORY_FAMILY,
                                       FILM_IN_CATEGORY_FOREIGN,
                                       FILM_IN_CATEGORY_GAMES,
                                       FILM_IN_CATEGORY_HORROR,
                                       FILM_IN_CATEGORY_MUSIC,
                                       FILM_IN_CATEGORY_NEW,
                                       FILM_IN_CATEGORY_SCIFI,
                                       FILM_IN_CATEGORY_SPORTS,
                                       FILM_IN_CATEGORY_TRAVEL)
    (SELECT F.FILM_ID,
            F.LAST_UPDATE,
            F.TITLE,
            F.DESCRIPTION,
            F.RELEASE_YEAR,
            L.NAME,
            F.RENTAL_DURATION                                  AS                     FILM_RENTAL_DURATION,
            F.RENTAL_RATE                                      AS                     FILM_RENTAL_RATE,
            F.LENGTH                                           AS                     FILM_DURATION,
            F.REPLACEMENT_COST                                 AS                     FILM_REPLACEMENT_COST,
            F.RATING                                           AS                     FILM_RATING_CODE,
            F.SPECIAL_FEATURES                                 AS                     FILM_RATING_TEXT,
            CASE WHEN F.SPECIAL_FEATURES LIKE '%Commentaries%' THEN 1 ELSE 0 END      FILM_HAS_COMMENTARIES,
            CASE WHEN F.SPECIAL_FEATURES LIKE '%Trailers%' THEN 1 ELSE 0 END          FILM_HAS_TRAILERS,
            CASE WHEN F.SPECIAL_FEATURES LIKE '%Deleted Scenes%' THEN 1 ELSE 0 END    FILM_HAS_DELETED_SCENES,
            CASE WHEN F.SPECIAL_FEATURES LIKE '%Behind the Scenes%' THEN 1 ELSE 0 END FILM_HAS_BEHIND_THE_SCENES,
            CASE WHEN C.NAME = 'Action' THEN 1 ELSE 0 END      AS                     FILM_IN_CATEGORY_ACTION,
            CASE WHEN C.NAME = 'Animation' THEN 1 ELSE 0 END   AS                     FILM_IN_CATEGORY_ANIMATION,
            CASE WHEN C.NAME = 'Children' THEN 1 ELSE 0 END    AS                     FILM_IN_CATEGORY_CHILDREN,
            CASE WHEN C.NAME = 'Classics' THEN 1 ELSE 0 END    AS                     FILM_IN_CATEGORY_CLASSICS,
            CASE WHEN C.NAME = 'Comedy' THEN 1 ELSE 0 END      AS                     FILM_IN_CATEGORY_COMEDY,
            CASE WHEN C.NAME = 'Documentary' THEN 1 ELSE 0 END AS                     FILM_IN_CATEGORY_DOCUMENTARY,
            CASE WHEN C.NAME = 'Drama' THEN 1 ELSE 0 END       AS                     FILM_IN_CATEGORY_DRAMA,
            CASE WHEN C.NAME = 'Family' THEN 1 ELSE 0 END      AS                     FILM_IN_CATEGORY_FAMILY,
            CASE WHEN C.NAME = 'Foreign' THEN 1 ELSE 0 END     AS                     FILM_IN_CATEGORY_FOREIGN,
            CASE WHEN C.NAME = 'Games' THEN 1 ELSE 0 END       AS                     FILM_IN_CATEGORY_GAMES,
            CASE WHEN C.NAME = 'Horror' THEN 1 ELSE 0 END      AS                     FILM_IN_CATEGORY_HORROR,
            CASE WHEN C.NAME = 'Music' THEN 1 ELSE 0 END       AS                     FILM_IN_CATEGORY_MUSIC,
            CASE WHEN C.NAME = 'New' THEN 1 ELSE 0 END         AS                     FILM_IN_CATEGORY_NEW,
            CASE WHEN C.NAME = 'Sci-Fi' THEN 1 ELSE 0 END      AS                     FILM_IN_CATEGORY_SCIFI,
            CASE WHEN C.NAME = 'Sports' THEN 1 ELSE 0 END      AS                     FILM_IN_CATEGORY_SPORTS,
            CASE WHEN C.NAME = 'Travel' THEN 1 ELSE 0 END      AS                     FILM_IN_CATEGORY_TRAVEL
     FROM SAKILA.CATEGORY C,
          SAKILA.FILM F,
          SAKILA.FILM_CATEGORY FC,
          SAKILA.LANGUAGE L
     WHERE F.FILM_ID = FC.FILM_ID
       AND F.LANGUAGE_ID = L.LANGUAGE_ID
       AND C.CATEGORY_ID = FC.CATEGORY_ID);


# dim_actor_bridge
INSERT INTO SAKILA_SNOWFLAKE.DIM_FILM_ACTOR_BRIDGE (FILM_KEY,
                                                    ACTOR_KEY)
    (SELECT FILM_KEY,
            ACTOR_KEY
     FROM SAKILA.FILM_ACTOR S_FA,
          DIM_ACTOR D_A,
          DIM_FILM D_F
     WHERE S_FA.ACTOR_ID = D_A.ACTOR_ID
       AND S_FA.FILM_ID = D_F.FILM_ID);


# The below query might take over 30 seconds to complete and you might get an "Error Code: 2013. 
# Lost connection to MySQL server during query" error
# Please follow the instructions below:
#   - In the application menu, select Edit > Preferences > SQL Editor.
#   - Look for the MySQL Session section and increase the DBMS connection read time out value.
#   - Save the settings, quit MySQL Workbench and reopen the connection.

-- -----------------------------------------------------
-- Write Fact table fact_rental DML script here
-- -----------------------------------------------------

INSERT INTO SAKILA_SNOWFLAKE.FACT_RENTAL (RENTAL_LAST_UPDATE, CUSTOMER_KEY, STAFF_KEY, FILM_KEY, STORE_KEY,
                                          RENTAL_DATE_KEY, RETURN_DATE_KEY, COUNT_RENTALS, COUNT_RETURNS,
                                          RENTAL_DURATION, DOLLAR_AMOUNT)
    (SELECT R.LAST_UPDATE,
            C.CUSTOMER_KEY,
            S.STAFF_KEY,
            F.FILM_KEY,
            ST.STORE_KEY,
            D1.DATE_KEY,
            D2.DATE_KEY,
            COUNT(R.RENTAL_ID),
            CASE WHEN R.RETURN_DATE IS NOT NULL THEN COUNT(R.RENTAL_ID) ELSE 0 END,
            F.FILM_RENTAL_DURATION,
            P.AMOUNT
     FROM SAKILA.RENTAL AS R
              INNER JOIN SAKILA.INVENTORY AS I ON R.INVENTORY_ID = I.INVENTORY_ID
              INNER JOIN SAKILA_SNOWFLAKE.DIM_CUSTOMER AS C ON C.CUSTOMER_ID = R.CUSTOMER_ID
              INNER JOIN SAKILA_SNOWFLAKE.DIM_STAFF AS S ON S.STAFF_ID = R.STAFF_ID
              INNER JOIN SAKILA_SNOWFLAKE.DIM_FILM AS F ON F.FILM_ID = I.FILM_ID
              INNER JOIN SAKILA_SNOWFLAKE.DIM_STORE AS ST ON ST.STORE_ID = I.STORE_ID
              INNER JOIN SAKILA.PAYMENT AS P ON P.RENTAL_ID = R.RENTAL_ID
              INNER JOIN SAKILA_SNOWFLAKE.DIM_DATE AS D1 ON D1.DATE = DATE(R.RENTAL_DATE)
              LEFT JOIN SAKILA_SNOWFLAKE.DIM_DATE AS D2 ON D2.DATE = DATE(R.RETURN_DATE)
     GROUP BY R.RENTAL_ID);
