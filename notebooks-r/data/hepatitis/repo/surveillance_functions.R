######### FUNCTIONS FOR SURVEILLANCE PLOTS #############

# Author: EMMANUELLE A. DANKWA (email:dankwa@stats.ox.ac.uk)

######################################################



## FUNCTION TO PLOT OBSERVED CASE COUNTS BY RISK ###
# Input:
#'@dat2_yrwk: Data frame of observed incidence by year, week and risk group


plot.by.risk <- function(dat2_yrwk){
  
  ## Prepare data ##
  
  # Create empty matrix to hold input values for plot
  values <- matrix(NA, nrow = 2, ncol = 91)
  
  # Fill in values 
  values[1,] <- dat2_yrwk$targetcases        # target
  values[2,] <- dat2_yrwk$non_target_cases   # non-target
  
  # create positions for tick marks, one more than number of bars
  at_tick <- seq_len(ncol(values) + 1)
  
  # Generating dates by week between the 36th week of 2017 and the 22nd week of 2019.
  
  wk_labels <- seq(as.Date("2017-09-03"), as.Date("2019-05-26"), by="week")
  wk_labels <- format(strptime(as.character(wk_labels),  format = "%Y-%m-%d" ), "%d/%m/%y")
  
  # Decrease number of dates labelled 
  chosen <- wk_labels[seq(1, 91, 5)]
  final_labels <- bespoke_insertion(wk_labels, chosen)
  
  # Now, plot
  par(mar = c(6, 4, 4, 4))   # par(mar = c(bottom, left, top, right))
  
  windowsFonts(A = windowsFont("Times New Roman")) # Set font
  par(family = "serif", cex= 1.1)
  barplot(values,  col = c('indianred3', 'grey'), las = 2, ylab= ' ', xlab = '', main = '', ylim = c(0, 35), space = 0 , axes = FALSE)
  legend(50, 30, legend = c('PEH/PWUD', 'Other'), fill = c('indianred3', 'grey'), title = 'Risk group', cex= 1.3)
  axis(side = 2, las = 1, pos = 0)
  axis(side = 1, at = at_tick - 1 , labels = FALSE)
  #axis(side = 1, at = seq_along(vacc_counts_target2$Count) - 0.5 , tick = FALSE, labels = c(1:22), las=2, srt=45 )
  text(x = seq_along(values[2,]) - 4.0, y = -1.7,  labels = final_labels, pos = 1, xpd = T, srt=45, cex = 1.2)
  title(xlab = 'Week', line = 4.5, cex.lab = 1.5)
  title(ylab = 'Number of detected cases', line = 2, cex.lab = 1.5)
  title(main = 'A', adj = 0, cex = 2.0)
  # Text indicating vaccination start week 
  text(x = which(final_labels == "21/01/18"), y = 30, labels = "Start of \n vaccinations", cex = 1.5) # Vaccination start date: 23/01/18; that is, week of 21/1/18
  arrows(x0 =  which(final_labels == "21/01/18"), y0 = 28, y1 = 15, lwd = 2, angle = 30 )
  }




## FUNCTION TO PLOT VACCINATION COUNTS ###
# Input:
#'@vacc_counts_target2: Data frame of vaccination counts by year and week among risk group

plot.vaccinations <- function(vacc_counts_target2){
  
  at_tick <- seq_len(length(vacc_counts_target2$Count) + 1)
  
  #Generating dates by week between the 4th and 25th weeks of 2018
  wk_labels <- seq(as.Date("2018-01-21"), as.Date("2018-06-17"), by="week") # Dates used here represent the beginning of the week for dates of interest
  wk_labels <- format(strptime(as.character(wk_labels),  format = "%Y-%m-%d" ), "%d/%m/%y")
  
  # Set margins
  par(mar = c(7, 4, 4, 2))   # par(mar = c(bottom, left, top, right))
  barplot(vacc_counts_target2$Count,  col = 'deepskyblue4' , ylab= ' ', xlab = '', main = '', space = 0 , axes = FALSE, ylim = c(0, 2000))
  axis(side = 2, las = 1, pos = 0)
  axis(side = 1, at = at_tick - 1, labels = FALSE)
  text(x = seq_along(vacc_counts_target2$Count) - 1.2,y = -100,  par("usr")[3] - 0.5, labels = wk_labels, pos = 1, xpd = T, srt=45, cex = 1.2)
  title(xlab = 'Week', line = 5 , cex.lab = 1.5)
  title(ylab = 'Number of vaccinations among PEH/PWUD', line = 2.5, cex.lab = 1.5)
  title(main = 'B', adj = 0, cex = 2.0 )
}  



# Plot vaccination counts on a comparable axis as the case counts 


plot.vaccinations.comparable.axis <- function(vacc_counts_target2){
  
  vacc <- c(rep(0, 20), vacc_counts_target2$Count, rep(0, 49))
  
  at_tick <- seq_len(length(vacc) + 1)
  
  # Generating dates by week 
  wk_labels <- seq(as.Date("2017-09-03"), as.Date("2019-05-26"), by="week")
  wk_labels <- format(strptime(as.character(wk_labels),  format = "%Y-%m-%d" ), "%d/%m/%y")
  
  
  # Decrease number of dates labelled 
  windowsFonts(A = windowsFont("Times New Roman")) # Set font
  par(family = "serif", cex= 1.1)
  chosen <- wk_labels[seq(1, 91, 5)]
  final_labels <- bespoke_insertion(wk_labels, chosen)
  # Now, plot
  
  
  # Set margins
  par(mar = c(7, 4, 4, 2))   # par(mar = c(bottom, left, top, right))
  barplot(vacc , col = 'deepskyblue4' , ylab= ' ', xlab = '', main = '', space = 0 , axes = FALSE, ylim = c(0, 2000))
  axis(side = 2, las = 1, pos = 0)
  axis(side = 1, at = at_tick - 1, labels = FALSE)
  text(x = seq_along(vacc) - 3.5,y = -120,  par("usr")[3] - 0.5, labels = final_labels, pos = 1, xpd = T, srt=45, cex = 1.2)
  title(xlab = 'Week', line = 5 , cex.lab = 1.5)
  title(ylab = 'Number of vaccinations among PEH/PWUD', line = 2.5, cex.lab = 1.5)
  title(main = '', adj = 0, cex = 2.0 )
  title(main = 'B', adj = 0, cex = 2.0 )
}








## FUNCTION TO INSERT WEEK LABELS PARTICULAR POSITIONS ###
# Input: 
#'@wk_labels: vector of week labels 
#'@chosen: chosen locations

bespoke_insertion <- function(wk_labels, chosen){
  
  new_wk_labels <- c()
  
  for (i in 1:length(wk_labels)){
    
    if(wk_labels[i] %in% chosen){
      
      new_wk_labels[i] <- wk_labels[i]
    }else{
      
      new_wk_labels[i] <- NA
    }
    
  }
  
  return(new_wk_labels)
  
}




## FUNCTION TO COMPUTE PERCENTAGES FOR SUMMARY TABLE ###
# Input: 
#'@x: vector of numbers to be converted to percentages
#'@total: total number of cases


perc <- function(x, total = 501){
  
  perc <- c()      # Vector to save percentages rounded to one decimal place
  perc.2 <- c()    # Vector to save percentages rounded to two decimal places
  lgl <- all.equal(sum(x), total)
  
  for (k in 1:length(x)){
    perc[k] <- round((x[k]/total)*100, 1)
  }
  lgl.perc <- all.equal(sum(perc), 100)
  if(lgl.perc != TRUE){
    for (k in 1:length(x)){
      perc.2[k] <- round((x[k]/total)*100,2)
    }
  }
  if(lgl.perc == TRUE){
    return(list(Do.components.add.up.to.total. = lgl, Rounded.percentages.to.one.dp = perc, Sum.rounded.percentages.to.one.dp = sum(perc)))
  }else{
    return(list(Do.components.add.up.to.total. = lgl, Rounded.percentages.to.one.dp = perc, Sum.rounded.percentages.to.one.dp = sum(perc),   Percentages.rounded.to.two.dp = perc.2, Sum.rounded.percentages.to.two.dp = sum(perc.2)))
  }
  
}



## ## FUNCTION TO COMPUTE PERCENTAGES FOR SUMMARY TABLE ###
# Input: 
#'@col: Variable for which summaries are to be computed 
#'@dat: Data
#'@hosp.total: Total number of hospitalizations
#'@mort.total: Total number of deaths


generate.summaries <- function(d_c, 
                               dat = dat2,
                               hosp.total = sum(dat$Hospitalization== 'Yes'),
                               mort.total = sum(dat$Mortality== 'Yes')){
  
  ls <-  list()  # Empty list to save output values 
  
  # Counts
  ls$cases <- summary(as.factor(d_c))        # Cases
  # Pick up position of variable
  ls$hosp <- table(d_c, dat[, 6])[, "Yes"]  # Hospitalizations
  ls$mort <- table(d_c, dat[, 7])[, "Yes"]  # Mortality
  
  # Percentages
  ls$cases_perc <- perc(as.numeric(ls$cases))[[2]]            # % cases
  ls$hosp_perc <- perc(as.numeric(ls$hosp), hosp.total)[[2]] # % hospitalizations
  ls$mort_perc <- perc(as.numeric(ls$mort), mort.total)[[2]] # % mortality
  return(ls)
}



