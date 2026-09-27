# Shiny version of the re-analysis page.   shiny::runApp("reanalysis/R")
if (!isTRUE(l10n_info()[["UTF-8"]])) invisible(Sys.setlocale("LC_CTYPE", "C.UTF-8"))
library(shiny)
library(bslib)
library(jsonlite)

root <- if (file.exists("web/data/netflix.json")) "." else ".."
D <- fromJSON(file.path(root, "web", "data", "netflix.json"), simplifyVector = FALSE)
F <- D$films
titles <- unlist(F$t); n <- unlist(F$n); m <- unlist(F$m)
yr <- vapply(F$y, function(v) if (is.null(v)) NA_integer_ else as.integer(v), 1L)
res <- do.call(rbind, lapply(D$metrics$results, function(r) data.frame(split = r$split, global_mean = r$global_mean,
                                                                       movie_mean = r$movie_mean, biases = r$biases, mf = r$mf)))
ord <- order(-n)

ui <- page_navbar(
  title = "Netflix Prize, revisited",
  theme = bs_theme(version = 5, primary = "#b0272f", base_font = "IBM Plex Sans, system-ui"),
  nav_panel("Benzer filmler / Similar films",
            selectizeInput("film", NULL, choices = setNames(ord, sprintf("%s (%s)", titles[ord], yr[ord])),
                           selected = ord[grep("Fellowship of the Ring", titles[ord])[1]], width = "100%"),
            textOutput("info"), tableOutput("nb")),
  nav_panel("Sonuçlar / Results", tableOutput("res"), plotOutput("curve", height = "320px"))
)

server <- function(input, output, session) {
  i <- reactive(as.integer(input$film))
  output$info <- renderText(sprintf("%s ratings · mean %.2f", format(n[i()], big.mark = ","), m[i()]))
  output$nb <- renderTable({
    nb <- F$nb[[i()]]
    j <- vapply(nb, function(x) x[[1]], 1) + 1; s <- vapply(nb, function(x) x[[2]], 1)
    data.frame(film = titles[j], year = yr[j], similarity = sprintf("%.0f%%", 100 * s), ratings = n[j], mean = m[j])
  })
  output$res <- renderTable(res, digits = 4)
  output$curve <- renderPlot({
    cv <- do.call(rbind, lapply(D$curve, as.data.frame))
    plot(test_rmse ~ epoch, cv[cv$split == "probe", ], type = "l", lwd = 2, col = "#b0272f", ylim = range(cv$test_rmse), ylab = "test RMSE")
    lines(test_rmse ~ epoch, cv[cv$split == "random", ], lwd = 2, col = "grey50")
    legend("topright", c("probe", "random 20%"), col = c("#b0272f", "grey50"), lwd = 2, bty = "n")
  })
}

shinyApp(ui, server)
