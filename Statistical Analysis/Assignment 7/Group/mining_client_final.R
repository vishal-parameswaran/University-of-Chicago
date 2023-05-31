source("Mining_connection.R")
library(nloptr)


# Read in logistic model --------------------------------------------------

model <- readRDS("model.rds")

# Define regression function ----------------------------------------------

regression <- function(tc, logRate, regressionTimes, returnError) {
  logregressionTimes <- log(tc - regressionTimes)
  linModel <- lm(logRate ~ logregressionTimes)

  if (returnError) {
    err <- sqrt(mean(linModel$residuals^2))
    res <- err
  } else {
    p <- linModel$coefficients[[2]]
    names(p) <- NULL
    res <- p
  }
  return(res)
}

# global vars
incoming_signals_counter <- 0 # incoming signals event counter
outgoing_signals_counter <- 0 # outgoing signals event counter
BUF_SIZE <- 10000 # we create buffers in advance:
inc_signals <- data.frame(time = .POSIXct(rep(NA, BUF_SIZE))) # dataframe for incoming signals
out_signals <- data.frame(time = .POSIXct(rep(NA, BUF_SIZE))) # dataframe for outgoing signals

eventMoments <- rep(NA, BUF_SIZE) # time spans from i-th signal and the first signal in minutes, eventMoments[1] = 0
initial_timestamp <- Sys.time() # start-of-program timestamp
plot_timestamp <- initial_timestamp # we use plot_timestamp to draw the plot once in a second

# Parameters for the simple solution
dt <- 0.05
w <- 10 # Experiment with window width to estimate intensity
n <- 50 # Experiment with number of observations of intensity in the model
t0 <- ((n + w) * dt) # window size for intensity estimating
eventRate_barrier <- 0.45 # when intensity exceeds this barrier then we send alarm signal!

# user defined handler
## no arguments
## returns:
#### logical vector of unit length which value is TRUE when we want to send signal back to server
## !! Note that only the first outgoing signal makes sense! Others will be ignored by server.

# This simple code is given as an example only. You can tune "eventRate_barrier" parameter for the
# workshop data, but it won't give you the best result on the test data.
# We will send an alarm signal if estimated event rate exceeds eventRate_barrier.
new_event_handler <- function() {
  now <- Sys.time()
  if (incoming_signals_counter < 0.5) {
    initial_timestamp <<- now
  }
  # time in minutes from the start of the stream
  t <- as.double(difftime(now, initial_timestamp, unit = "min"))
  # log event if necessary
  message("EVENT at ", now)

  # update inc_signals dataframe (append last value):
  incoming_signals_counter <<- incoming_signals_counter + 1
  inc_signals[incoming_signals_counter, ] <<- list(now)
  eventMoments[incoming_signals_counter] <<- t

  send_signal <- FALSE
  if ((t > t0) & (outgoing_signals_counter <= 0)) {
    tGrid <- seq(from = t - t0 + dt, to = t, by = dt)
    X <- eventMoments[!is.na(eventMoments)] # This is 'events' in the workshop
    eventsGrid <- findInterval(tGrid, X)

    N <- length(tGrid)
    # Intensity
    intensity <- eventsGrid[(w + 1):N] - eventsGrid[1:(N - w)]
    # Events per minute
    intensity <- intensity / (dt * w)
    timeGrid <- tGrid[(N - n + 1):N]
    logIntensity <- log(pmax(intensity, 0.1))

    res <- nloptr(
      # Vector with starting values for the optimization
      x0 = t + 1,
      # Function that returns the value of the objective function
      eval_f = regression,
      # Lower bounds of the controls
      lb = t + 0.1,
      # Upper bounds
      ub = t + 10,
      # Some options
      opts = list("algorithm" = "NLOPT_LN_COBYLA", "xtol_rel" = 1e-04),
      # Arguments passed to user-defined objective function
      logRate = logIntensity,
      regressionTimes = timeGrid,
      returnError = TRUE
    )

    # Parameters
    p <- regression(res$solution, logIntensity, timeGrid, FALSE)
    time_to_shock <- res$solution - t
    intensities <- intensity[[length(intensity)]]

    # Predict
    prob <- predict(
      object = model,
      newdata = data.frame(
        pp = p,
        intensities = intensities,
        time_to_shock = time_to_shock
      ),
      type = "response"
    )

    message("Predicted Probablity is: ", prob, " \nThreshold is ", eventRate_barrier)

    send_signal <- (prob > eventRate_barrier)
  }

  if (send_signal) {
    # update out_signals dataframe (append last value):
    outgoing_signals_counter <<- outgoing_signals_counter + 1
    out_signals[outgoing_signals_counter, ] <<- list(now)
  }

  Draw()

  return(send_signal)
}


# plot once in a second
Draw <- function() {
  now <- Sys.time()
  if (difftime(now, plot_timestamp, unit = "sec") >= 1) {
    plot_timestamp <<- now
    if (incoming_signals_counter > 0) {
      t <- difftime(inc_signals$time[1:incoming_signals_counter], initial_timestamp, unit = "min")
      plot(
        x = t, y = 1:length(t),
        xlim = c(0, difftime(now, initial_timestamp, unit = "min")),
        type = "s", xlab = "time (minutes)", ylab = "n_events"
      )

      if (outgoing_signals_counter > 0) {
        # draw outgoing signal (only the first one)
        abline(
          v = difftime(out_signals$time, initial_timestamp, unit = "min")[1],
          col = "red", lwd = 2
        )
      }
    }
  }
}


# server options
host <- "datastream.ilykei.com"
port <- 30011
login <- "vishalp@uchicago.edu"
password <- "qR22GtzM"
stream_name <- "mining_statistics"
catch_handler_errors <- TRUE # we recommend using TRUE during the test and FALSE during homework
# make connection with your personal handler
result <- Connect(host, port, login, password, stream_name, new_event_handler, catch_handler_errors)

# remove empty values from buffers
inc_signals <- inc_signals[!is.na(inc_signals$time), ]
out_signals <- out_signals[!is.na(out_signals$time), ]
eventMoments <- eventMoments[1:incoming_signals_counter]
alarmTime <- as.double(difftime(out_signals[1], inc_signals[1], unit = "min"))
message("alarmTime = ", alarmTime)

# after all you can dump your data/results and analyze it later
dump(c("inc_signals", "out_signals", "result"), file = "results.txt")
