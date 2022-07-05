
library(shiny)
library(dplyr)
library(StatMatch)
library(shinydashboard)
#if (interactive()) {
#  library(shiny)
#  library(shinydashboard)
#  library(shinydashboardPlus)
# Cargamos los datos necesarios

path <- 'C:/Users/marcl/Desktop/TFG/GITHUB TFG/data/'
datos <- read.csv2(paste0(path, 'HotelesImputados.csv'))

vars <- c("Hotel", "Estrellas", "Aceptan.Mascotas", "Aire.Acondicionado", "Aparcamiento.Disponible",
          "Aparcamiento.Gratis", "Cocina.Básica", "Desayuno.Gratis", "Gimnasio", 
          "Para.No.Fumadores", "Piscina", "Traslado.Al.Aeropuerto", "Wifi.Gratis", "precios",
          "ratioDescr")

vars2 <- c("Estrellas", "Aceptan.Mascotas", "Aire.Acondicionado", "Aparcamiento.Disponible",
           "Aparcamiento.Gratis", "Cocina.Básica", "Desayuno.Gratis", "Gimnasio", 
           "Para.No.Fumadores", "Piscina", "Traslado.Al.Aeropuerto", "Wifi.Gratis", "precios",
           "ratioDescr")

vars3 <- c("Aceptan.Mascotas", "Aire.Acondicionado", "Aparcamiento.Disponible",
           "Aparcamiento.Gratis", "Cocina.Básica", "Desayuno.Gratis", "Gimnasio", 
           "Para.No.Fumadores", "Piscina", "Traslado.Al.Aeropuerto", "Wifi.Gratis")

vars4 <- c("Hotel", "Estrellas", "Aceptan.Mascotas", "Aire.Acondicionado", "Aparcamiento.Disponible",
           "Aparcamiento.Gratis", "Cocina.Básica", "Desayuno.Gratis", "Gimnasio", 
           "Para.No.Fumadores", "Piscina", "Traslado.Al.Aeropuerto", "Wifi.Gratis", "precios",
           "ratioDescr")

#vars[which(!vars %in% colnames(datos))]

datos <- datos[, vars]
datos$precios <- as.numeric(datos$precios)

for (var in vars3) {
    datos[, var] <- ifelse(datos[, var] == 1, "Si", "No")
}


# Define UI for application that draws a histogram
ui <- fluidPage(
    
    # Sidebar with a slider input for number of bins 
    sidebarLayout(
        sidebarPanel(
            fluidRow(
                column(4, 
                    radioButtons("mascotas", label = h3("Aceptan Mascotas"),
                                 choices = list("No" = "No", "Si" = "Si"), 
                                 selected = "No"),
                    radioButtons("aire", label = h3("Aire Acondicionado"),
                                 choices = list("No" = "No", "Si" = "Si"), 
                                 selected = "No"),
                    radioButtons("aparDispo", label = h3("Aparcamiento Disponible"),
                                 choices = list("No" = "No", "Si" = "Si"), 
                                 selected = "No"),
                    radioButtons("aparGratis", label = h3("Aparcamiento Gratis"),
                                 choices = list("No" = "No", "Si" = "Si"), 
                                 selected = "No"),
                ),
                column(4,
                   radioButtons("cocina", label = h3("Cocina"),
                                choices = list("No" = "No", "Si" = "Si"), 
                                selected = "No"),
                   radioButtons("desayuno", label = h3("Desayunos gratis"),
                                choices = list("No" = "No", "Si" = "Si"), 
                                selected = "No"),
                    radioButtons("gym", label = h3("Gimnasio"),
                                 choices = list("No" = "No", "Si" = "Si"), 
                                 selected = "No"),
                    radioButtons("fumadores", label = h3("Fumadores"),
                                 choices = list("No" = "No", "Si" = "Si"), 
                                 selected = "No"),
                ),
                column(4, 
                    radioButtons("piscina", label = h3("Piscina"),
                                 choices = list("No" = "No", "Si" = "Si"), 
                                 selected = "No"),
                    radioButtons("aero", label = h3("Traslado al Aeropuerto"),
                                 choices = list("No" = "No", "Si" = "Si"), 
                                 selected = "No"),
                    radioButtons("wifi", label = h3("Wifi"),
                                 choices = list("No" = "No", "Si" = "Si"), 
                                 selected = "No"),
                    radioButtons("coments", label = h3("Comentaris"),
                                 choices = list("Bueno" = "Bueno", 
                                                "Excepcional" = "Excepcional", 
                                                "Fabuloso" = "Fabuloso", 
                                                "Increíble" = "Increíble", 
                                                "Muy Bueno" = "Muy Bueno"), 
                                 selected = "Bueno"),
                ),
            ),
                sliderInput("estrellas", label = h3("Estrellas"), min = 0, 
                            max = 5, value = 2),
                
                sliderInput("precio", label = h3("Precio"), min = 0, 
                            max = 2000, value = 100)
        ),

        # Show a plot of the generated distribution
        mainPanel(
           sliderInput("n", label = h3("Top N"), min = 1, 
                       max = 20, value = 5),
           h1("Características del hotel"),
           tableOutput("taulaOpcio"),
           hr(),
           h1("Hoteles recomendados"),
           tableOutput("taulaRecomendados")
        )
    )
)

# Define server logic required to draw a histogram
server <- function(input, output) {
    datosEscogidos <- reactive({
        data <- data.frame("Estrellas" = as.character(input$estrellas), 
                           "Aceptan.Mascotas" = input$mascotas, 
                           "Aire.Acondicionado" = input$aire, 
                           "Aparcamiento.Disponible" = input$aparDispo,
                           "Aparcamiento.Gratis" = input$aparGratis, 
                           "Cocina.Básica" = input$cocina, 
                           "Desayuno.Gratis" = input$desayuno, 
                           "Gimnasio" = input$gym, 
                           "Para.No.Fumadores" = input$fumadores, 
                           "Piscina" = input$piscina,
                           "Traslado.Al.Aeropuerto" = input$aero, 
                           "Wifi.Gratis" = input$wifi, 
                           "precios" = input$precio,
                           "ratioDescr" = input$coments)
        data <- data[, vars2]
        return(data)
    })
    
    datosRecomender <- reactive({
        matriz <- anti_join(datos[, vars2], datosEscogidos())
        matriz <- data.frame(distinct(matriz))
        matriz2 <- datos[, vars4]
        v <- datosEscogidos()
        matrisfin <- rbind(v, matriz)
        print(matrisfin)
        
        dists <- gower.dist(data.x = matrisfin[1, ], 
                            data.y = matrisfin[-1, ])^2 ##
        matriz3 <- matriz2[order(dists)[1:input$n], ] 
        print(matriz3)
        return(matriz3)
    })
    
    
    output$taulaOpcio <- renderTable(datosEscogidos()) 
    output$taulaRecomendados <- renderTable(datosRecomender()) 

}

shinyApp(ui = ui, server = server)

