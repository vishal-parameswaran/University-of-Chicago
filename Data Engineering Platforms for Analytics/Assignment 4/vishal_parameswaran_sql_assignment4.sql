########################## ASSIGNMENT 4a SQL ##############################

# Name: Vishal Parameswaran
# Date: 21/10/2022

####### INSTRUCTIONS #######

# Read through the whole template and read each question carefully.  Make sure to follow all instructions.

# Each question should be answered with only one SQL query per question, unless otherwise stated.
# All queries must be written below the corresponding question number.
# Make sure to include the schema name in all table references (i.e. sakila.customer, not just customer)
# DO NOT modify the comment text for each question unless asked.
# Any additional comments you may wish to add to organize your code MUST be on their own lines and each comment line must begin with a # character
# If a question asks for specific columns and/or column aliases, they MUST be followed.
# Pay attention to the requested column aliases for aggregations and calculations. Otherwise, do not re-alias columns from the original column names in the tables unless asked to do so.
# Return columns in the order requested in the question.
# Do not concatenate columns together unless asked.

# Refer to the Sakila documentation for further information about the tables, views, and columns: https://dev.mysql.com/doc/sakila/en/

##########################################################################

## Desc: Joining Data, Nested Queries, Views and Indexes, Transforming Data

############################ PREREQUESITES ###############################

# These queries make use of the Sakila schema.  If you have issues with the Sakila schema, try dropping the schema and re-loading it from the scripts provided with Assignment 2.

# Run the following two SQL statements before beginning the questions:
SET SQL_SAFE_UPDATES = 0;
UPDATE SAKILA.FILM
SET LANGUAGE_ID=6
WHERE TITLE LIKE '%Academy%';

############################### QUESTION 1 ###############################

# 1a) List the actors (first_name, last_name, actor_id) who acted in more then 25 movies.  Also return the count of movies they acted in, aliased as movie_count. Order by first and last name alphabetically.

SELECT A.FIRST_NAME, A.LAST_NAME, A.ACTOR_ID, COUNT(FA.ACTOR_ID) AS MOVIE_COUNT
FROM SAKILA.ACTOR AS A
         INNER JOIN SAKILA.FILM_ACTOR AS FA ON A.ACTOR_ID = FA.ACTOR_ID
GROUP BY A.ACTOR_ID, A.FIRST_NAME, A.LAST_NAME
HAVING MOVIE_COUNT > 25
ORDER BY A.FIRST_NAME, A.LAST_NAME;


# 1b) List the actors (first_name, last_name, actor_id) who have worked in German language movies. Order by first and last name alphabetically.

SELECT A.FIRST_NAME, A.LAST_NAME, A.ACTOR_ID
FROM SAKILA.ACTOR AS A
         INNER JOIN SAKILA.FILM_ACTOR AS FA ON A.ACTOR_ID = FA.ACTOR_ID
WHERE FA.FILM_ID IN (SELECT F1.FILM_ID
                     FROM SAKILA.FILM F1
                     WHERE LANGUAGE_ID = (SELECT L.LANGUAGE_ID
                                          FROM SAKILA.LANGUAGE AS L
                                          WHERE L.NAME = 'German'))
ORDER BY A.FIRST_NAME, A.LAST_NAME;

# 1c) List the actors (first_name, last_name, actor_id) who acted in horror movies and the count of horror movies by each actor.  Alias the count column as horror_movie_count. Order by first and last name alphabetically.

SELECT A.FIRST_NAME, A.LAST_NAME, A.ACTOR_ID, COUNT(FA.ACTOR_ID) AS HORROR_MOVIE_COUNT
FROM SAKILA.ACTOR AS A
         INNER JOIN SAKILA.FILM_ACTOR AS FA ON A.ACTOR_ID = FA.ACTOR_ID
WHERE FA.FILM_ID IN (SELECT F1.FILM_ID
                     FROM SAKILA.FILM_CATEGORY F1
                     WHERE CATEGORY_ID = (SELECT C.CATEGORY_ID
                                          FROM SAKILA.CATEGORY AS C
                                          WHERE C.NAME = 'Horror'))
GROUP BY A.ACTOR_ID, A.FIRST_NAME, A.LAST_NAME
ORDER BY A.FIRST_NAME, A.LAST_NAME;

# 1d) List the customers who rented more than 3 horror movies.  Return the customer first and last names, customer IDs, and the horror movie rental count (aliased as horror_movie_count). Order by first and last name alphabetically.

SELECT C.FIRST_NAME, C.LAST_NAME, C.CUSTOMER_ID, COUNT(C.CUSTOMER_ID) AS HORROR_MOVIE_COUNT
FROM SAKILA.CUSTOMER AS C
         INNER JOIN SAKILA.RENTAL AS R ON C.CUSTOMER_ID = R.CUSTOMER_ID
         INNER JOIN SAKILA.INVENTORY AS I ON R.INVENTORY_ID = I.INVENTORY_ID
WHERE I.FILM_ID IN (SELECT F1.FILM_ID
                    FROM SAKILA.FILM_CATEGORY F1
                    WHERE CATEGORY_ID = (SELECT C.CATEGORY_ID
                                         FROM SAKILA.CATEGORY AS C
                                         WHERE C.NAME = 'Horror'))
GROUP BY C.FIRST_NAME, C.LAST_NAME, C.CUSTOMER_ID
HAVING HORROR_MOVIE_COUNT > 3
ORDER BY C.FIRST_NAME, C.LAST_NAME;


# 1e) List the customers who rented a movie which starred Scarlett Bening.  Return the customer first and last names and customer IDs. Order by first and last name alphabetically.

SELECT C.FIRST_NAME, C.LAST_NAME, C.CUSTOMER_ID
FROM SAKILA.CUSTOMER AS C
         INNER JOIN SAKILA.RENTAL AS R ON C.CUSTOMER_ID = R.CUSTOMER_ID
         INNER JOIN (SELECT I1.INVENTORY_ID
                     FROM SAKILA.INVENTORY AS I1
                     WHERE I1.FILM_ID IN (SELECT FA.FILM_ID
                                          FROM SAKILA.FILM_ACTOR AS FA
                                          WHERE FA.ACTOR_ID IN (SELECT A.ACTOR_ID
                                                                FROM SAKILA.ACTOR AS A
                                                                WHERE A.FIRST_NAME = 'Scarlett'
                                                                  AND A.LAST_NAME = 'Bening'))) AS I
                    ON I.INVENTORY_ID = R.INVENTORY_ID
GROUP BY C.FIRST_NAME, C.LAST_NAME, C.CUSTOMER_ID
ORDER BY C.FIRST_NAME, C.LAST_NAME;


SELECT C.FIRST_NAME, C.LAST_NAME, C.CUSTOMER_ID
FROM SAKILA.CUSTOMER AS C
         INNER JOIN SAKILA.RENTAL AS R ON C.CUSTOMER_ID = R.CUSTOMER_ID
         INNER JOIN SAKILA.INVENTORY AS I ON I.INVENTORY_ID = R.INVENTORY_ID
         INNER JOIN SAKILA.FILM_ACTOR AS FA on I.FILM_ID = FA.FILM_ID
         INNER JOIN SAKILA.ACTOR as A on FA.ACTOR_ID = A.ACTOR_ID
where A.FIRST_NAME = 'Scarlett'
  AND A.LAST_NAME = 'Bening'
GROUP BY C.FIRST_NAME, C.LAST_NAME, C.CUSTOMER_ID
ORDER BY C.FIRST_NAME, C.LAST_NAME;


# 1f) Which customers residing at postal code 62703 rented movies that were Documentaries?  Return their first and last names and customer IDs.  Order by first and last name alphabetically.
SELECT C.FIRST_NAME, C.LAST_NAME, C.CUSTOMER_ID
FROM SAKILA.CUSTOMER AS C
         INNER JOIN SAKILA.RENTAL AS R ON C.CUSTOMER_ID = R.CUSTOMER_ID
         INNER JOIN SAKILA.INVENTORY AS I ON R.INVENTORY_ID = I.INVENTORY_ID
WHERE I.FILM_ID IN (SELECT F1.FILM_ID
                    FROM SAKILA.FILM_CATEGORY F1
                    WHERE CATEGORY_ID = (SELECT C.CATEGORY_ID
                                         FROM SAKILA.CATEGORY AS C
                                         WHERE C.NAME = 'Documentary'))
  AND C.ADDRESS_ID IN (SELECT A.ADDRESS_ID FROM SAKILA.ADDRESS AS A WHERE A.POSTAL_CODE = '62703')
GROUP BY C.FIRST_NAME, C.LAST_NAME, C.CUSTOMER_ID
ORDER BY C.FIRST_NAME, C.LAST_NAME;

# 1g) Find all the addresses (if any) where the second address line is not empty and is not NULL (i.e., contains some text).  Return the address_id and address_2, sorted by address_2 in ascending order.

SELECT A.ADDRESS_ID, A.ADDRESS2
FROM SAKILA.ADDRESS AS A
WHERE A.ADDRESS2 IS NOT NULL
  AND A.ADDRESS2 <> ''
ORDER BY A.ADDRESS2;

# 1h) List the actors (first_name, last_name, actor_id)  who played in a film involving a “Crocodile” and a “Shark” (in the same movie).  Also include the title and release year of the movie.  Sort the results by the actors’ last name then first name, in ascending order.

SELECT A.FIRST_NAME, A.LAST_NAME, A.ACTOR_ID, F.TITLE, F.RELEASE_YEAR
FROM SAKILA.ACTOR AS A
         INNER JOIN SAKILA.FILM_ACTOR AS FA ON A.ACTOR_ID = FA.ACTOR_ID
         INNER JOIN (SELECT F1.FILM_ID, F1.TITLE, F1.RELEASE_YEAR
                     FROM SAKILA.FILM AS F1
                     WHERE F1.DESCRIPTION LIKE '%Crocodile%'
                       AND F1.DESCRIPTION LIKE '%Shark%') AS F
                    ON FA.FILM_ID = F.FILM_ID
ORDER BY A.LAST_NAME, A.FIRST_NAME;

# 1i) Find all the film categories in which there are between 55 and 65 films. Return the category names and the count of films per category, sorted from highest to lowest by the number of films.  Alias the count column as count_movies. Order the results alphabetically by category.

SELECT C.NAME, COUNT(C.CATEGORY_ID) AS COUNT_MOVIES
FROM SAKILA.CATEGORY AS C
         INNER JOIN SAKILA.FILM_CATEGORY AS FC ON C.CATEGORY_ID = FC.CATEGORY_ID
GROUP BY C.CATEGORY_ID, C.NAME
HAVING COUNT_MOVIES BETWEEN 55 AND 65
ORDER BY COUNT_MOVIES DESC, C.NAME;

# 1j) In which of the film categories is the average difference between the film replacement cost and the rental rate larger than $17?  Return the film categories and the average cost difference, aliased as mean_diff_replace_rental.  Order the results alphabetically by category.

SELECT C.NAME, AVG(F.REPLACEMENT_COST - F.RENTAL_RATE) AS MEAN_DIFF_REPLACE_RENTAL
FROM SAKILA.CATEGORY AS C
         INNER JOIN SAKILA.FILM_CATEGORY AS FC ON C.CATEGORY_ID = FC.CATEGORY_ID
         INNER JOIN SAKILA.FILM AS F ON FC.FILM_ID = F.FILM_ID
GROUP BY C.CATEGORY_ID, C.NAME
HAVING MEAN_DIFF_REPLACE_RENTAL > 17
ORDER BY C.NAME;

# 1k) Create a list of overdue rentals so that customers can be contacted and asked to return their overdue DVDs. Return the title of the  film, the customer first name and last name, customer phone number, and the number of days overdue, aliased as days_overdue.  Order the results by first and last name alphabetically.
## NOTE: To identify if a rental is overdue, find rentals that have not been returned and have a rental date rental date further in the past than the film's rental duration (rental duration is in days)

SELECT F.TITLE,
       C.FIRST_NAME,
       C.LAST_NAME,
       (SELECT A.PHONE FROM SAKILA.ADDRESS AS A WHERE A.ADDRESS_ID = C.ADDRESS_ID) AS PHONE_NUMBER,
       DATEDIFF(CURRENT_DATE(), R.RENTAL_DATE) - F.RENTAL_DURATION                 AS DAYS_OVERDUE
FROM SAKILA.CUSTOMER AS C
         INNER JOIN (SELECT R1.CUSTOMER_ID, R1.INVENTORY_ID, R1.LAST_UPDATE, R1.RENTAL_DATE
                     FROM SAKILA.RENTAL AS R1
                     WHERE R1.RETURN_DATE IS NULL) AS R ON C.CUSTOMER_ID = R.CUSTOMER_ID
         INNER JOIN SAKILA.INVENTORY AS I ON R.INVENTORY_ID = I.INVENTORY_ID
         INNER JOIN SAKILA.FILM F ON I.FILM_ID = F.FILM_ID
WHERE DATEDIFF(CURRENT_DATE(), R.RENTAL_DATE) > F.RENTAL_DURATION
ORDER BY C.FIRST_NAME, C.LAST_NAME;

# 1l) Find the list of all customers and staff for store_id 1.  Return the first and last names, as well as a column indicating if the name is 'staff' or 'customer', aliased as person_type.  Order results by first name and last name alphabetically.
## Note : use a set operator and do not remove duplicates

SELECT C.FIRST_NAME, C.LAST_NAME, 'Customer' AS PERSON_TYPE
FROM SAKILA.CUSTOMER AS C
WHERE C.STORE_ID = 1
UNION
SELECT ST.FIRST_NAME, ST.LAST_NAME, 'Staff' AS PERSON_TYPE
FROM SAKILA.STAFF AS ST
WHERE ST.STORE_ID = 1
ORDER BY 1, 2;
############################### QUESTION 2 ###############################

# 2a) List the first and last names of both actors and customers whose first names are the same as the first name of the actor with actor_id 8.  Order in alphabetical order by last name.
## NOTE: Do not remove duplicates and do not hard-code the first name in your query.

SELECT K.FIRST_NAME, K.LAST_NAME
FROM (SELECT C.FIRST_NAME, C.LAST_NAME
      FROM SAKILA.CUSTOMER AS C
      UNION
      SELECT A.FIRST_NAME, A.LAST_NAME
      FROM SAKILA.ACTOR AS A) AS K
WHERE K.FIRST_NAME = (SELECT A1.FIRST_NAME FROM SAKILA.ACTOR AS A1 WHERE A1.ACTOR_ID = 8)
ORDER BY K.LAST_NAME;

# 2b) List customers (first name, last name, customer ID) and payment amounts of customer payments that were greater than average the payment amount.  Sort in descending order by payment amount.
## HINT: Use a subquery to help

SELECT C.FIRST_NAME, C.LAST_NAME, C.CUSTOMER_ID, P.AMOUNT AS PAYMENT_AMOUNT
FROM SAKILA.CUSTOMER AS C
         INNER JOIN SAKILA.PAYMENT AS P ON C.CUSTOMER_ID = P.CUSTOMER_ID
WHERE P.AMOUNT > (SELECT AVG(P1.AMOUNT) FROM SAKILA.PAYMENT AS P1)
ORDER BY PAYMENT_AMOUNT DESC;

# 2c) List customers (first name, last name, customer ID) who have rented movies at least once.  Order results by first name, lastname alphabetically.
## Note: use an IN clause with a subquery to filter customers

SELECT C.FIRST_NAME, C.LAST_NAME, C.CUSTOMER_ID
FROM SAKILA.CUSTOMER AS C
WHERE C.CUSTOMER_ID IN (SELECT DISTINCT(R.CUSTOMER_ID) FROM SAKILA.RENTAL AS R)
ORDER BY C.FIRST_NAME, C.LAST_NAME;

# 2d) Find the floor of the maximum, minimum and average payment amount.  Alias the result columns as max_payment, min_payment, avg_payment.

SELECT FLOOR(MAX(P.AMOUNT)) AS MAX_PAYMENT,
       FLOOR(MIN(P.AMOUNT)) AS MIN_PAYMENT,
       FLOOR(AVG(P.AMOUNT)) AS AVG_PAYMENT
FROM SAKILA.PAYMENT AS P;


############################### QUESTION 3 ###############################

# 3a) Create a view called actors_portfolio which contains the following columns of information about actors and their films: actor_id, first_name, last_name, film_id, title, category_id, category_name
CREATE VIEW SAKILA.ACTORS_PORTFOLIO AS
SELECT FA.ACTOR_ID, A.FIRST_NAME, A.LAST_NAME, F.FILM_ID, F.TITLE, FC.CATEGORY_ID, C.NAME as CATEGORY_NAME
FROM SAKILA.ACTOR AS A
         INNER JOIN SAKILA.FILM_ACTOR FA ON A.ACTOR_ID = FA.ACTOR_ID
         INNER JOIN SAKILA.FILM F ON FA.FILM_ID = F.FILM_ID
         INNER JOIN SAKILA.FILM_CATEGORY FC ON F.FILM_ID = FC.FILM_ID
         INNER JOIN SAKILA.CATEGORY C ON FC.CATEGORY_ID = C.CATEGORY_ID;

# 3b) Describe (using a SQL command) the structure of the view.

DESCRIBE SAKILA.ACTORS_PORTFOLIO;

# 3c) Query the view to get information (all columns) on the actor ADAM GRANT

SELECT *
FROM SAKILA.ACTORS_PORTFOLIO AS AP
WHERE AP.FIRST_NAME = 'Adam'
  AND AP.LAST_NAME = 'Grant';

# 3d) Insert a new movie titled Data Hero in Sci-Fi Category starring ADAM GRANT
## NOTE: If you need to use multiple statements for this question, you may do so.
## WARNING: Do not hard-code any id numbers in your where criteria.
## !! Think about how you might do this before reading the hints below !!
## HINT 1: Given what you know about a view, can you insert directly into the view? Or do you need to insert the data elsewhere?
## HINT 2: Consider using SET and LAST_INSERT_ID() to set a variable to aid in your process.
SET @LANGUAGE_ID = (SELECT L.LANGUAGE_ID
                    FROM SAKILA.LANGUAGE AS L
                    WHERE L.NAME = 'English');
SET @ACTOR_ID = (SELECT A.ACTOR_ID
                 FROM SAKILA.ACTOR AS A
                 WHERE A.FIRST_NAME = 'Adam'
                   AND A.LAST_NAME = 'Grant');
SET @SCI_FI_CATEGORY_ID = (SELECT C.CATEGORY_ID
                           FROM SAKILA.CATEGORY AS C
                           WHERE C.NAME = 'Sci-Fi');
INSERT INTO SAKILA.FILM (TITLE, LANGUAGE_ID)
VALUES ('Data Hero', @LANGUAGE_ID);
SET @LAST_ID_IN_FILM = LAST_INSERT_ID();
INSERT INTO SAKILA.FILM_ACTOR (ACTOR_ID, FILM_ID)
VALUES (@ACTOR_ID, @LAST_ID_IN_FILM);
INSERT INTO SAKILA.FILM_CATEGORY(FILM_ID, CATEGORY_ID)
VALUES (@LAST_ID_IN_FILM, @SCI_FI_CATEGORY_ID);
############################### QUESTION 4 ###############################

# 4a) Extract the street number (numbers at the beginning of the address) from the customer address in the customer_list view.  Return the original address column, and the street number column (aliased as street_number).  Order the results in ascending order by street number.
## NOTE: Use Regex to parse the street number

SELECT CL.ADDRESS, REGEXP_SUBSTR(CL.ADDRESS, '([^ ]+)') AS STREET_NUMBER
FROM SAKILA.CUSTOMER_LIST AS CL
ORDER BY CAST(STREET_NUMBER AS UNSIGNED);

# 4b) List actors (first name, last name, actor id) whose last name starts with characters A, B or C.  Order by first_name, last_name in ascending order.
## NOTE: Use either a LEFT() or RIGHT() operator

SELECT A.FIRST_NAME, A.LAST_NAME, A.ACTOR_ID
FROM SAKILA.ACTOR AS A
WHERE LEFT(A.LAST_NAME, 1) IN ('A', 'B', 'C')
ORDER BY A.FIRST_NAME, A.LAST_NAME;

# 4c) List film titles that contain exactly 10 characters.  Order titles in ascending alphabetical order.

SELECT F.TITLE
FROM SAKILA.FILM AS F
WHERE LENGTH(F.TITLE) = 10
ORDER BY F.TITLE;

# 4d) Return a list of distinct payment dates formatted in the date pattern that matches '22/01/2016' (2 digit day, 2 digit month, 4 digit year).  Alias the formatted column as payment_date.  Retrn the formatted dates in ascending order.

SELECT DISTINCT DATE_FORMAT(P.PAYMENT_DATE, '%d/%m/%Y') AS PAYMENT_DATE
FROM SAKILA.PAYMENT AS P
ORDER BY P.PAYMENT_DATE;

# 4e) Find the number of days each rental was out (days between rental_date & return_date), for all returned rentals.  Return the rental_id, rental_date, return_date, and alias the days between column as days_out.  Order with the longest number of days_out first.

SELECT R.RENTAL_ID, R.RENTAL_DATE, R.RETURN_DATE, DATEDIFF(R.RETURN_DATE, R.RENTAL_DATE) AS DAYS_OUT
FROM SAKILA.RENTAL AS R
WHERE RETURN_DATE IS NOT NULL
ORDER BY DAYS_OUT DESC;