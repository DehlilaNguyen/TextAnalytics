library(rvest) #static content
library(RSelenium) #dynamic content
library(tidyverse)
library(netstat)

imdb_link <- "https://www.imdb.com/search/title/?title_type=feature&year=2023-01-01,2023-12-31"
imdb_page <- read_html(imdb_link)
imdb_page %>%
  html_nodes(".lister-item-header a") %>%
  html_text()

movie_urls <- imdb_page %>%
  html_nodes(".lister-item-header a") %>%
  html_attr("href")

movie_info <- data.frame(movie_title = movie_titles,
                         url = movie_urls)
rm(list(movie_titles, movie_urls))

finviz_stock_url <- "https://finviz.com/quote.ashx?t=TSLA&p=d"
finviz_page <- read_html(finviz_stock_url)
tsla_stock_data <- finviz_page %>%
  html_nodes(".snapshot-table-wrapper") %>%
  html_table()
tsla_stock_data <- as.data.frame(tsla_stock_data[[1]])



remotedr <- rsDriver(
  port = netstat::free_port(),
  browser = c("chrome"),
  chromever = "114.0.5735.90",
  verbose = F
)



