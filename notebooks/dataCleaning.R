## Loading R packages and source the "getshots" customized own function
library(jsonlite)
library(tidyverse)
library(ggsoccer)
library(dplyr)
library(REdaS)
library(yd2m)
library(purrr)

##################### The first dataset ##############################

# code and data from https://github.com/Dato-Futbol/xg-model
get_shots <- function(file_path, name_detail, save_files = F){

  players <- fromJSON("data/players.json")

  shots <- fromJSON(file_path) %>%
    filter(subEventName == "Shot")

  tags <- tibble(tags = shots$tags) %>%
    hoist(tags,
          tags_id = "id") %>%
    unnest_wider(tags_id, names_sep = "")

  tags2 <- tags %>%
    mutate(is_goal = ifelse(rowSums(. == "101", na.rm = T) > 0, 1, 0),
           is_blocked = ifelse(rowSums(. == "2101", na.rm = T) > 0, 1, 0),
           is_CA = ifelse(rowSums(. == "1901", na.rm = T) > 0, 1, 0), # is countre attack
           body_part = ifelse(rowSums(. == "401", na.rm = T) > 0, "left",
                              ifelse(rowSums(. == "402", na.rm = T) > 0, "right",
                                     ifelse(rowSums(. == "403", na.rm = T) > 0, "head/body", "NA"))))

  pos <- tibble(positions = shots$positions) %>%
    hoist(positions,
          y = "y",
          x = "x") %>%
    unnest_wider(y, names_sep = "") %>%
    unnest_wider(x, names_sep = "") %>%
    dplyr::select(-c(x2, y2))

  shots_ok <- shots %>%
    dplyr::select(matchId, teamId, playerId, eventSec, matchPeriod) %>%
    bind_cols(pos, tags2) %>%
    filter(is_blocked == 0) %>%
    dplyr::select(-c(8:13)) %>%
    left_join(players %>%
                dplyr::select(c("wyId", "foot")), by = c("playerId" = "wyId")) %>%
    mutate(league = name_detail)

  if(save_files){
    write_rds(shots, paste0("shots", name_detail, ".rds"))
    write_rds(tags2, paste0("tags2", name_detail, ".rds"))
    write_rds(pos, paste0("pos", name_detail, ".rds"))
    write_rds(shots_ok, paste0("unblocked_shots", name_detail, ".rds"))
  }

  shots_ok
}
# shotsEN <- get_shots("data/events/events_England.json", "EN")
# shotsSP <- get_shots("data/events/events_Spain.json", "SP")
# shotsWC <- get_shots("data/events/events_World_Cup.json", "WC")
# shotsIT <- get_shots("data/events/events_Italy.json", "IT")
# shotsGE <- get_shots("data/events/events_Germany.json", "GE")
# shotsFR <- get_shots("data/events/events_France.json", "FR")
# shotsEC <- get_shots("data/events/events_European_Championship.json", "EC")
#
# shots <- shotsEN %>%
#   bind_rows(shotsFR, shotsGE, shotsIT, shotsSP, shotsWC, shotsEC)

get_final_data <- function(data) {

  data <- data %>% select(eventSec, y1, x1, is_goal, is_blocked, is_CA, body_part, foot)
  data$x1 <- (100 - data$x1) * 105/100
  data$y1 <- data$y1 * data$y1/100
  data <- data %>% mutate(angle = atan(7.32 * x1 / (x1^2 + y1^2 - (7.32/2)^2)))
  data$angle <- ifelse(data$angle<0, base::pi + data$angle, data$angle)
  data <- data %>% mutate(distance = sqrt( (100 - x1)^2 + (34 - y1)^2),
                          minute = round(eventSec / 60),
                          eventSec = round(eventSec))

  data
}

# data1 <- get_final_data(shots)
# write.csv(data1, file = "data/data1.csv")

##################### The second dataset ##############################

get_data <- function(event_path, info_path) {
  events <- read.csv(event_path)
  info <- read.csv(info_path)

  events <- merge(events, info[, c('id_odsp', 'country', 'date')], by = 'id_odsp', all.x = TRUE)
  data <- subset(events, event_type == 1)

  data_final <- data %>% select(sort_order, time, shot_place, shot_outcome, is_goal, location, bodypart, assist_method, situation,
                                fast_break)
  data_final

}

# data2 <- get_data(event_path = "data/events.csv", info_path = "data/ginf.csv")
# write.csv(data2, file = "data/data2.csv")

##################### The third dataset ##############################

# make angle from the x, y coordinates for the 3rd dataset
loc2angle <- function(x, y) {
  rads <- atan(7.32 * x / (x^2 + (y - 34)^2 - (7.32/2)^2))
  rads <- ifelse(rads<0, base::pi + rads, rads)
  deg <- rad2deg(rads)
  deg
}

# distance to goal
loc2distance <- function(x, y) {
  sqrt(x^2 + (y - 34)^2)
}

# distance between two points on the pitch
loc2locdistance <- function(x1, y1, x2, y2) {
  sqrt( (x1 - x2)^2 + (y1 - y2)^2 )
}

get_shots2 <- function(json_file) {
  data <- fromJSON(json_file) %>% filter(type$name == "Shot") %>% dplyr::select(c(minute, position, location, shot))

  df_temp <- do.call(rbind, lapply(data$location, function(loc) c(120, 80) - loc))
  colnames(df_temp) <- c("x1", "y1")

  data$x1 <- df_temp[,1]
  data$y1 <- df_temp[,2]

  data$shot$freeze_frame <- Map(function(ff, x1, y1) {
    ff$x1 <- yd_to_m(x1)
    ff$y1 <- yd_to_m(y1)
    return(ff)
  },
  data$shot$freeze_frame, data$x1, data$y1)

  tryCatch({
    df_players_location <- mapply( function(sublist) {
      if (!is.null(sublist$teammate)) {
        df_players <- sapply(sublist$location, function(loc) c(120, 80) - loc %>% as.numeric() %>% yd_to_m() %>% round(., digits = 1)) %>% t() %>% as.data.frame()
        # df <- sapply(sublist$teammate, function(tmt) cbind(df_players, tmt))
        df <- cbind(df_players, sublist$teammate, sublist$position$name, sublist$x1, sublist$y1)
        colnames(df) <- c("x", "y", "teammate", "position_name", "x1", "y1")
        df <- df %>% mutate(teammate = ifelse(teammate, "teammate", "opponent"),
                            distance = loc2locdistance(x1 = x, y1 = y, x2 = x1, y2 = y1)) %>%
                     arrange(distance)

        groups_count <- df %>% group_by(teammate) %>% count() %>% as.data.frame()
        if ( !("opponent" %in% groups_count$teammate) ) {
          groups_count <- groups_count %>% add_row(teammate = "opponent", n = 0)
        } else if ( !("teammate" %in% groups_count$teammate) ) {
          groups_count <- groups_count %>% add_row(teammate = "teammate", n = 0)
        }

        na_df <- as.data.frame(matrix("na", nrow = 21 - nrow(df), ncol = ncol(df)))
        colnames(na_df) <- colnames(df)

        na_df$teammate <- rep(c("opponent", "teammate"), c(11, 10) - groups_count$n)
        dff <- rbind(df, na_df)
        dff <- dff %>% group_by(teammate) %>% mutate(rown = row_number(distance)) %>% ungroup() %>%
          mutate(position_teammate = paste(teammate, ifelse(position_name == "Goalkeeper", position_name, rown), sep = "_")) %>%
          select(-c(teammate, position_name, rown, distance, x1, y1)) %>%
          mutate(x = ifelse(x == "na", NA, x),
                 y = ifelse(x == "na", NA, y))
      } else {
        dff <- as.data.frame(matrix("na", nrow = 21, ncol = 3))
        colnames(dff) <- c("x", "y", "teammate")
        dff$teammate <- rep(c("opponent", "teammate"), c(11, 10))
        dff <- dff %>% group_by(teammate) %>% mutate(rown = row_number()) %>% ungroup() %>%
          mutate(position_teammate = paste(teammate, rown, sep = "_")) %>%
          select(-c(teammate, rown)) %>%
          mutate(x = ifelse(x == "na", NA, x),
                 y = ifelse(x == "na", NA, y))
      }
      # print(wider_df)
      # stop("123")
      # %>%
      # stop("123")
      wider_df <- dff %>%
        pivot_wider(names_from = position_teammate, values_from = c(x, y), names_sep = "_player_") %>%
        mutate(across(everything(), as.numeric))
      wider_df
      # wider_df <- apply(wider_df, MARGIN = 2, unlist)
    }, data$shot$freeze_frame)
  },
    error = function(e) {
      # handle the error
      print(json_file)
      print(paste("An error occurred:", e$message))
  })
  df_players_location <- df_players_location %>% t()

  tryCatch({ # TODO reduce error cases
    data$number_of_players_opponents <- mapply(function(sublist, x1_threshold) {
      # Extracting the first location value and converting it to numeric
      first_location_values <- sapply(sublist$location, function(loc) as.numeric(loc[1]))

      if ("teammate" %in% names(sublist)) {
      # Filtering and counting
        res <- sum(!sublist$teammate & first_location_values > x1_threshold) # error here
      } else {
        res <- 0
      }
      res
    }, data$shot$freeze_frame, data$x1)
  },
  error = function(e) {
    print(json_file)
    # handle the error
    print(paste("An error occurred:", e$message))
  })

  tryCatch({ # TODO reduce error cases
    data$number_of_players_teammates <- mapply(function(sublist, x1_threshold) {
      # Extracting the first location value and converting it to numeric
      first_location_values <- sapply(sublist$location, function(loc) as.numeric(loc[1]))

      if ("teammate" %in% names(sublist)) {
        # Filtering and counting
        res <- sum(sublist$teammate & first_location_values > x1_threshold) # error here
      } else {
        res <- 0
      }
      res
    }, data$shot$freeze_frame, data$x1)
  },
  error = function(e) {
    print(json_file)
    # handle the error
    print(paste("An error occurred:", e$message))
  })


  data$shot <- data$shot %>% select(-freeze_frame, -statsbomb_xg, -key_pass_id)
  data$shot$body_part <- data$shot$body_part %>% select(-id)
  data$shot$technique <- data$shot$technique %>% select(-id)
  data$shot$type <- data$shot$type %>% select(-id)
  data$position <- data$position %>% select(-id)

  data$shot <- data$shot %>% select(-end_location)

  tryCatch({ # TODO reduce error cases
    if ("one_on_one" %in% colnames(data$shot)) {
      data[is.na(data$shot$one_on_one), ]$shot$one_on_one <- FALSE
    } else {
      data$shot$one_on_one <- FALSE
    }

    if ("first_time" %in% colnames(data$shot)) {
      data[is.na(data$shot$first_time), ]$shot$first_time <- FALSE
    } else {
      data$shot$first_time <- FALSE
    }

    if ("aerial_won" %in% colnames(data$shot)) {
      data[is.na(data$shot$aerial_won), ]$shot$aerial_won <- FALSE
    } else {
      data$shot$aerial_won <- FALSE
    }

    if ("saved_to_post" %in% colnames(data$shot)) {
      data[is.na(data$shot$saved_to_post), ]$shot$saved_to_post <- FALSE
    } else {
      data$shot$saved_to_post <- FALSE
    }

    if ("deflected" %in% colnames(data$shot)) {
      data[is.na(data$shot$deflected), ]$shot$deflected <- FALSE
    } else {
      data$shot$deflected <- FALSE
    }

    if ("saved_off_target" %in% colnames(data$shot)) {
      data[is.na(data$shot$saved_off_target), ]$shot$saved_off_target <- FALSE
    } else {
      data$shot$saved_off_target <- FALSE
    }

    if ("open_goal" %in% colnames(data$shot)) {
      data[is.na(data$shot$open_goal), ]$shot$open_goal <- FALSE
    } else {
      data$shot$open_goal <- FALSE
    }

    if ("follows_dribble" %in% colnames(data$shot)) {
      data[is.na(data$shot$follows_dribble), ]$shot$follows_dribble <- FALSE
    } else {
      data$shot$follows_dribble <- FALSE
    }

    if ("redirect" %in% colnames(data$shot)) {
      data[is.na(data$shot$redirect), ]$shot$redirect <- FALSE
    } else {
      data$shot$redirect <- FALSE
    }

    if ("kick_off" %in% colnames(data$kick_off)) {
      data[is.na(data$shot$kick_off), ]$shotf$kick_off <- FALSE
    } else {
      data$kick_off <- FALSE
    }
  },
  error = function(e) {
    # handle the error
    print(paste("An error occurred:", e$message))
  })

  data <- data %>% mutate(is_goal = ifelse(shot$outcome$id == 97, 1, 0),
                          x1 = yd_to_m(x1) %>% round(., digits = 1),
                          y1 = yd_to_m(y1) %>% round(., digits = 1),
                          angle = loc2angle(x1, y1) %>% round(., digits = 1),
                          distance = loc2distance(x = x1, y = y1)) %>%
                  select(-location)
  data$shot$outcome <- data$shot$outcome %>% select(-id)
  data <- data %>% unnest(shot, names_sep = "_") %>%
    unnest(position, names_sep = "_") %>%
    unnest(shot_type, names_sep = "_") %>%
    unnest(shot_outcome, names_sep = "_") %>%
    unnest(shot_technique, names_sep = "_") %>%
    unnest(shot_body_part, names_sep = "_")

  data <- cbind(data, df_players_location)
  data
}

file_names <- list.files(path = "data/la_liga_events/", pattern = "*.json")
data_list <- lapply(paste("data/la_liga_events/", file_names, sep = ""), get_shots2)
combined_data <- bind_rows(data_list)
skimr::skim(combined_data)

# # sample data
# data <- fromJSON("data/la_liga_events/ (1000).json") %>% filter(type$name == "Shot") %>% dplyr::select(c(minute, position, location, shot))

data3_final <- combined_data %>% select(-c(shot_outcome_name,
                                   shot_saved_off_target,
                                   shot_saved_to_post,
                                   kick_off)) %>%
  mutate(shot_kick_off = ifelse(is.na(shot_kick_off), FALSE, shot_kick_off))
pattern <- "^(x_player_|y_player_).*$"
cols <- names(data3_final)[grepl(pattern, names(data3_final))]
data_final <- data3_final %>% unnest(all_of(cols))
skimr::skim(data_final)
write_csv(data_final, file = "data/final_data.csv")
#df_test <- read.csv("data/final_data.csv", nrows = 10000)
##################### The fourth dataset ##############################
