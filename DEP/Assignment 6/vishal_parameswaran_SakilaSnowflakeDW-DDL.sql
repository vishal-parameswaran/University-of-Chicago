/***********************************************
**                MSc ANALYTICS 
**     DATA ENGINEERING PLATFORMS (MSCA 31012)
** File:   Sakila Snowflake DDL - Assignment 5
** Desc:   Creating the Sakila Snowflake Dimensional model
** Auth:   Shreenidhi Bharadwaj, Ashish Pujari, Audrey Salerno
** Date:   04/08/2018, Last updated 02/09/2021
************************************************/

SET @OLD_UNIQUE_CHECKS = @@UNIQUE_CHECKS, UNIQUE_CHECKS = 0;
SET @OLD_FOREIGN_KEY_CHECKS = @@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS = 0;
SET @OLD_SQL_MODE = @@SQL_MODE, SQL_MODE = 'TRADITIONAL,ALLOW_INVALID_DATES';

-- -----------------------------------------------------
-- Schema mydb
-- -----------------------------------------------------
-- -----------------------------------------------------
-- Schema sakila_snowflake
-- -----------------------------------------------------

CREATE SCHEMA IF NOT EXISTS `Sakila_Snowflake` DEFAULT CHARACTER SET LATIN1;
USE `Sakila_Snowflake`;

-- -----------------------------------------------------
-- Table `sakila_snowflake`.`dim_actor`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `Sakila_Snowflake`.`Dim_Actor`
(
    `Actor_Key`         INT(10)     NOT NULL AUTO_INCREMENT,
    `Actor_Last_Update` TIMESTAMP   NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    `Actor_Id`          INT(10)     NOT NULL,
    `Actor_Last_Name`   VARCHAR(45) NOT NULL,
    `Actor_First_Name`  VARCHAR(45) NOT NULL,
    PRIMARY KEY (`Actor_Key`)
)
    ENGINE = InnoDB
    DEFAULT CHARACTER SET = LATIN1;

CREATE INDEX `Dim_Actor_Last_Update` ON `Sakila_Snowflake`.`Dim_Actor` (`Actor_Last_Update` ASC);


-- -----------------------------------------------------
-- Table `sakila_snowflake`.`dim_location_country`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `Sakila_Snowflake`.`Dim_Location_Country`
(
    `Location_Country_Key`         INT(10)      NOT NULL AUTO_INCREMENT,
    `Location_Country_Id`          SMALLINT(10) NOT NULL,
    `Location_Country_Last_Update` TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    `Location_Country_Name`        VARCHAR(50)  NOT NULL,
    PRIMARY KEY (`Location_Country_Key`)
)
    ENGINE = InnoDB
    AUTO_INCREMENT = 110
    DEFAULT CHARACTER SET = LATIN1;


-- -----------------------------------------------------
-- Table `sakila_snowflake`.`dim_location_city`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `Sakila_Snowflake`.`Dim_Location_City`
(
    `Location_City_Key`         INT(10)      NOT NULL AUTO_INCREMENT,
    `Location_Country_Key`      INT(10)      NOT NULL,
    `Location_City_Name`        VARCHAR(50)  NOT NULL,
    `Location_City_Id`          SMALLINT(10) NOT NULL,
    `Location_City_Last_Update` TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (`Location_City_Key`),
    CONSTRAINT `Dim_Location_Country_Dim_Location_City_Fk`
        FOREIGN KEY (`Location_Country_Key`)
            REFERENCES `Sakila_Snowflake`.`Dim_Location_Country` (`Location_Country_Key`)
            ON DELETE NO ACTION
            ON UPDATE NO ACTION
)
    ENGINE = InnoDB
    AUTO_INCREMENT = 601
    DEFAULT CHARACTER SET = LATIN1;

CREATE INDEX `Dim_Location_Country_Dim_Location_City_Fk` ON `Sakila_Snowflake`.`Dim_Location_City` (`Location_Country_Key` ASC);


-- -----------------------------------------------------
-- Table `sakila_snowflake`.`dim_location_address`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `Sakila_Snowflake`.`Dim_Location_Address`
(
    `Location_Address_Key`         INT(10)     NOT NULL AUTO_INCREMENT,
    `Location_City_Key`            INT(10)     NOT NULL,
    `Location_Address_Id`          INT(10)     NOT NULL,
    `Location_Address_Last_Update` TIMESTAMP   NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    `Location_Address`             VARCHAR(64) NOT NULL,
    `Location_Address_Postal_Code` VARCHAR(10) NULL     DEFAULT NULL,
    PRIMARY KEY (`Location_Address_Key`),
    CONSTRAINT `Dim_Location_City_Dim_Location_Address_Fk`
        FOREIGN KEY (`Location_City_Key`)
            REFERENCES `Sakila_Snowflake`.`Dim_Location_City` (`Location_City_Key`)
            ON DELETE NO ACTION
            ON UPDATE NO ACTION
)
    ENGINE = InnoDB
    AUTO_INCREMENT = 604
    DEFAULT CHARACTER SET = LATIN1;

CREATE INDEX `Dim_Location_City_Dim_Location_Address_Fk` ON `Sakila_Snowflake`.`Dim_Location_Address` (`Location_City_Key` ASC);


-- -----------------------------------------------------
-- Table `sakila_snowflake`.`dim_customer`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `Sakila_Snowflake`.`Dim_Customer`
(
    `Customer_Key`            INT(8)      NOT NULL AUTO_INCREMENT,
    `Location_Address_Key`    INT(10)     NOT NULL,
    `Customer_Last_Update`    TIMESTAMP   NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    `Customer_Id`             INT(8)      NULL     DEFAULT NULL,
    `Customer_First_Name`     VARCHAR(45) NULL     DEFAULT NULL,
    `Customer_Last_Name`      VARCHAR(45) NULL     DEFAULT NULL,
    `Customer_Email`          VARCHAR(50) NULL     DEFAULT NULL,
    `Customer_Active`         CHAR(3)     NULL     DEFAULT NULL,
    `Customer_Created`        DATE        NULL     DEFAULT NULL,
    `Customer_Version_Number` SMALLINT(5) NULL     DEFAULT NULL,
    `Customer_Valid_From`     DATE        NULL     DEFAULT NULL,
    `Customer_Valid_Through`  DATE        NULL     DEFAULT NULL,
    PRIMARY KEY (`Customer_Key`),
    CONSTRAINT `Dim_Location_Address_Dim_Customer_Fk`
        FOREIGN KEY (`Location_Address_Key`)
            REFERENCES `Sakila_Snowflake`.`Dim_Location_Address` (`Location_Address_Key`)
            ON DELETE NO ACTION
            ON UPDATE NO ACTION
)
    ENGINE = InnoDB
    DEFAULT CHARACTER SET = LATIN1;

CREATE INDEX `Customer_Id` USING BTREE ON `Sakila_Snowflake`.`Dim_Customer` (`Customer_Id` ASC);
CREATE INDEX `Dim_Customer_Last_Update` ON `Sakila_Snowflake`.`Dim_Customer` (`Customer_Last_Update` ASC);
CREATE INDEX `Dim_Location_Address_Dim_Customer_Fk` ON `Sakila_Snowflake`.`Dim_Customer` (`Location_Address_Key` ASC);


-- -----------------------------------------------------
-- Table `sakila_snowflake`.`dim_film`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `Sakila_Snowflake`.`Dim_Film`
(
    `Film_Key`                     INT(8)        NOT NULL AUTO_INCREMENT,
    `Film_Last_Update`             TIMESTAMP     NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    `Film_Id`                      INT(10)       NOT NULL,
    `Film_Title`                   VARCHAR(64)   NOT NULL,
    `Film_Description`             TEXT                   DEFAULT NULL,
    `Film_Release_Year`            SMALLINT(5)            DEFAULT NULL,
    `Film_Language`                VARCHAR(20)   NOT NULL,
    `Film_Rental_Duration`         TINYINT(3)    NULL     DEFAULT NULL,
    `Film_Rental_Rate`             DECIMAL(4, 2) NULL     DEFAULT NULL,
    `Film_Duration`                INT(8)        NULL     DEFAULT NULL,
    `Film_Replacement_Cost`        DECIMAL(5, 2) NULL     DEFAULT NULL,
    `Film_Rating_Code`             CHAR(5)       NULL     DEFAULT NULL,
    `Film_Rating_Text`             VARCHAR(255)  NULL     DEFAULT NULL,
    `Film_Has_Trailers`            CHAR(4)       NULL     DEFAULT NULL,
    `Film_Has_Commentaries`        CHAR(4)       NULL     DEFAULT NULL,
    `Film_Has_Deleted_Scenes`      CHAR(4)       NULL     DEFAULT NULL,
    `Film_Has_Behind_The_Scenes`   CHAR(4)       NULL     DEFAULT NULL,
    `Film_In_Category_Action`      CHAR(4)       NULL     DEFAULT NULL,
    `Film_In_Category_Animation`   CHAR(4)       NULL     DEFAULT NULL,
    `Film_In_Category_Children`    CHAR(4)       NULL     DEFAULT NULL,
    `Film_In_Category_Classics`    CHAR(4)       NULL     DEFAULT NULL,
    `Film_In_Category_Comedy`      CHAR(4)       NULL     DEFAULT NULL,
    `Film_In_Category_Documentary` CHAR(4)       NULL     DEFAULT NULL,
    `Film_In_Category_Drama`       CHAR(4)       NULL     DEFAULT NULL,
    `Film_In_Category_Family`      CHAR(4)       NULL     DEFAULT NULL,
    `Film_In_Category_Foreign`     CHAR(4)       NULL     DEFAULT NULL,
    `Film_In_Category_Games`       CHAR(4)       NULL     DEFAULT NULL,
    `Film_In_Category_Horror`      CHAR(4)       NULL     DEFAULT NULL,
    `Film_In_Category_Music`       CHAR(4)       NULL     DEFAULT NULL,
    `Film_In_Category_New`         CHAR(4)       NULL     DEFAULT NULL,
    `Film_In_Category_Scifi`       CHAR(4)       NULL     DEFAULT NULL,
    `Film_In_Category_Sports`      CHAR(4)       NULL     DEFAULT NULL,
    `Film_In_Category_Travel`      CHAR(4)       NULL     DEFAULT NULL,
    PRIMARY KEY (`Film_Key`)
)
    ENGINE = InnoDB
    DEFAULT CHARACTER SET = LATIN1;

CREATE INDEX `Dim_Film_Last_Update` ON `Sakila_Snowflake`.`Dim_Film` (`Film_Last_Update` ASC);


-- -----------------------------------------------------
-- Table `sakila_snowflake`.`dim_film_actor_bridge`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `Sakila_Snowflake`.`Dim_Film_Actor_Bridge`
(
    `Film_Key`              INT(8)  NOT NULL,
    `Actor_Key`             INT(10) NOT NULL,
    `Actor_Weighing_Factor` DECIMAL(3, 2) DEFAULT NULL,
    PRIMARY KEY (`Film_Key`, `Actor_Key`),
    CONSTRAINT `Dim_Actor_Dim_Film_Actor_Bridge_Fk`
        FOREIGN KEY (`Actor_Key`)
            REFERENCES `Sakila_Snowflake`.`Dim_Actor` (`Actor_Key`)
            ON DELETE NO ACTION
            ON UPDATE NO ACTION,
    CONSTRAINT `Dim_Film_Dim_Film_Actor_Bridge_Fk`
        FOREIGN KEY (`Film_Key`)
            REFERENCES `Sakila_Snowflake`.`Dim_Film` (`Film_Key`)
            ON DELETE NO ACTION
            ON UPDATE NO ACTION
)
    ENGINE = InnoDB
    DEFAULT CHARACTER SET = LATIN1;

CREATE INDEX `Dim_Actor_Dim_Film_Actor_Bridge_Fk` ON `Sakila_Snowflake`.`Dim_Film_Actor_Bridge` (`Actor_Key` ASC);


-- -----------------------------------------------------
-- Table `sakila_snowflake`.`dim_staff`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `Sakila_Snowflake`.`Dim_Staff`
(
    `Staff_Key`            INT(8)      NOT NULL AUTO_INCREMENT,
    `Staff_Last_Update`    TIMESTAMP   NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    `Staff_Id`             INT(8)      NULL     DEFAULT NULL,
    `Staff_First_Name`     VARCHAR(45) NULL     DEFAULT NULL,
    `Staff_Last_Name`      VARCHAR(45) NULL     DEFAULT NULL,
    `Staff_Store_Id`       INT(8)      NULL     DEFAULT NULL,
    `Staff_Version_Number` SMALLINT(5) NULL     DEFAULT NULL,
    `Staff_Valid_From`     DATE        NULL     DEFAULT NULL,
    `Staff_Valid_Through`  DATE        NULL     DEFAULT NULL,
    `Staff_Active`         CHAR(3)     NULL     DEFAULT NULL,
    PRIMARY KEY (`Staff_Key`)
)
    ENGINE = InnoDB
    DEFAULT CHARACTER SET = LATIN1;

CREATE INDEX `Dim_Staff_Last_Update` ON `Sakila_Snowflake`.`Dim_Staff` (`Staff_Last_Update` ASC);


-- -----------------------------------------------------
-- Table `sakila_snowflake`.`dim_store`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `Sakila_Snowflake`.`Dim_Store`
(
    `Store_Key`                INT(8)      NOT NULL AUTO_INCREMENT,
    `Location_Address_Key`     INT(10)     NOT NULL,
    `Store_Last_Update`        TIME        NOT NULL,
    `Store_Id`                 INT(8)      NULL DEFAULT NULL,
    `Store_Manager_Staff_Id`   INT(8)      NULL DEFAULT NULL,
    `Store_Manager_First_Name` VARCHAR(45) NULL DEFAULT NULL,
    `Store_Manager_Last_Name`  VARCHAR(45) NULL DEFAULT NULL,
    `Store_Version_Number`     SMALLINT(5) NULL DEFAULT NULL,
    `Store_Valid_From`         DATE        NULL DEFAULT NULL,
    `Store_Valid_Through`      DATE        NULL DEFAULT NULL,
    PRIMARY KEY (`Store_Key`),
    CONSTRAINT `Dim_Location_Address_Dim_Store_Fk`
        FOREIGN KEY (`Location_Address_Key`)
            REFERENCES `Sakila_Snowflake`.`Dim_Location_Address` (`Location_Address_Key`)
            ON DELETE NO ACTION
            ON UPDATE NO ACTION
)
    ENGINE = InnoDB
    DEFAULT CHARACTER SET = LATIN1;

CREATE INDEX `Store_Id` USING BTREE ON `Sakila_Snowflake`.`Dim_Store` (`Store_Id` ASC);
CREATE INDEX `Dim_Store_Last_Update` ON `Sakila_Snowflake`.`Dim_Store` (`Store_Last_Update` ASC);
CREATE INDEX `Dim_Location_Address_Dim_Store_Fk` ON `Sakila_Snowflake`.`Dim_Store` (`Location_Address_Key` ASC);

-- -----------------------------------------------------
-- Table `sakila_snowflake`.`dim_date`
-- -----------------------------------------------------

CREATE TABLE IF NOT EXISTS `Sakila_Snowflake`.`Dim_Date`
(
    `Date_Key`             BIGINT(20) NOT NULL,
    `Date`                 DATE       NOT NULL,
    `Timestamp`            BIGINT(20) NULL     DEFAULT NULL,
    `Weekend`              CHAR(10)   NOT NULL DEFAULT 'Weekday',
    `Day_Of_Week`          CHAR(10)   NULL     DEFAULT NULL,
    `Month`                CHAR(10)   NULL     DEFAULT NULL,
    `Month_Day`            INT(11)    NULL     DEFAULT NULL,
    `Year`                 INT(11)    NULL     DEFAULT NULL,
    `Week_Starting_Monday` CHAR(2)    NULL     DEFAULT NULL,
    PRIMARY KEY (`Date_Key`),
    UNIQUE INDEX `Date` (`Date` ASC),
    INDEX `Year_Week` (`Year` ASC, `Week_Starting_Monday` ASC)
)
    ENGINE = InnoDB
    DEFAULT CHARACTER SET = LATIN1;

-- -----------------------------------------------------
-- Table `sakila_snowflake`.`numbers`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `Sakila_Snowflake`.`Numbers`
(
    `Number` BIGINT(20) NULL DEFAULT NULL
)
    ENGINE = InnoDB
    DEFAULT CHARACTER SET = LATIN1;

-- -----------------------------------------------------
-- Table `sakila_snowflake`.`numbers_small`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `Sakila_Snowflake`.`Numbers_Small`
(
    `Number` INT(11) NULL DEFAULT NULL
)
    ENGINE = InnoDB
    DEFAULT CHARACTER SET = LATIN1;


-- -----------------------------------------------------
-- Table `sakila_snowflake`.`fact_rental`
-- Write Fact table fact_rental DDL script here
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `Sakila_Snowflake`.`Fact_Rental`
(
    `Rental_Id`          INT(10) NOT NULL AUTO_INCREMENT,
    `Rental_Last_Update` TIMESTAMP NOT NULL ,
    `CUSTOMER_KEY`         INT(8) NOT NULL ,
    `STAFF_KEY`            INT(8) NOT NULL ,
    `FILM_KEY`             INT(8) NOT NULL ,
    `STORE_KEY`            INT(8) NOT NULL ,
    `RENTAL_DATE_KEY`      BIGINT(20)NOT NULL ,
    `RETURN_DATE_KEY`      BIGINT(20) DEFAULT NULL,
    `COUNT_RENTALS`        INT(8) NOT NULL ,
    `COUNT_RETURNS`        INT(8) NOT NULL ,
    `RENTAL_DURATION`      INT(10) NOT NULL ,
    `DOLLAR_AMOUNT`        FLOAT NOT NULL ,
    PRIMARY KEY (`Rental_Id`),
    CONSTRAINT `fk_Fact_Rental_Customer_Key` FOREIGN KEY (`Customer_Key`)
        REFERENCES `SAKILA_SNOWFLAKE`.`DIM_CUSTOMER` (`Customer_Key`)
        ON DELETE NO ACTION ON UPDATE NO ACTION,
    CONSTRAINT `fk_Fact_Rental_Staff_Key` FOREIGN KEY (`STAFF_KEY`)
        REFERENCES `SAKILA_SNOWFLAKE`.`DIM_STAFF` (`STAFF_KEY`)
        ON DELETE NO ACTION ON UPDATE NO ACTION,
    CONSTRAINT `fk_Fact_Rental_Film_Key` FOREIGN KEY (`FILM_KEY`)
        REFERENCES `SAKILA_SNOWFLAKE`.`DIM_FILM` (`FILM_KEY`)
        ON DELETE NO ACTION ON UPDATE NO ACTION,
    CONSTRAINT `fk_Fact_Rental_Store_Key` FOREIGN KEY (`STORE_KEY`)
        REFERENCES `SAKILA_SNOWFLAKE`.`DIM_STORE` (`STORE_KEY`)
        ON DELETE NO ACTION ON UPDATE NO ACTION,
    CONSTRAINT `fk_Fact_Rental_Rental_Date_Key` FOREIGN KEY (`RENTAL_DATE_KEY`)
        REFERENCES `SAKILA_SNOWFLAKE`.`DIM_DATE` (`DATE_KEY`)
        ON DELETE NO ACTION ON UPDATE NO ACTION,
    CONSTRAINT `fk_Fact_Rental_Return_Date_Key` FOREIGN KEY (`RETURN_DATE_KEY`)
        REFERENCES `SAKILA_SNOWFLAKE`.`DIM_DATE` (`DATE_KEY`)
        ON DELETE NO ACTION ON UPDATE NO ACTION
)
    ENGINE = InnoDB
    DEFAULT CHARACTER SET = LATIN1;



-- -----------------------------------------------------
-- Indexes  for `sakila_snowflake`.`fact_rental`
-- Run these after you create your table
-- -----------------------------------------------------

CREATE INDEX `Dim_Store_Fact_Rental_Fk` ON `Sakila_Snowflake`.`Fact_Rental` (`Store_Key` ASC);
CREATE INDEX `Dim_Staff_Fact_Rental_Fk` ON `Sakila_Snowflake`.`Fact_Rental` (`Staff_Key` ASC);
CREATE INDEX `Dim_Film_Fact_Rental_Fk` ON `Sakila_Snowflake`.`Fact_Rental` (`Film_Key` ASC);
CREATE INDEX `Dim_Customer_Fact_Rental_Fk` ON `Sakila_Snowflake`.`Fact_Rental` (`Customer_Key` ASC);

SET SQL_MODE = @OLD_SQL_MODE;
SET FOREIGN_KEY_CHECKS = @OLD_FOREIGN_KEY_CHECKS;
SET UNIQUE_CHECKS = @OLD_UNIQUE_CHECKS;
