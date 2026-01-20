// Detecto el scrolling para aplicar la animación de la barra de habilidades
let menuVisible = false;

// Función que oculta o muestra el menu
function mostrarOcultarMenu(){
    if(menuVisible){
        document.getElementById("nav").classList ="";
        menuVisible = false;
    }else{
        document.getElementById("nav").classList ="responsive";
        menuVisible = true;
    }
}

function seleccionar(){
    //oculto el menu una vez que selecciono una opcion
    document.getElementById("nav").classList = "";
    menuVisible = false;
}

// Funcion que aplica las animaciones de las habilidades
function efectoHabilidades(){
    var skills = document.getElementById("skills");
    var distancia_skills = window.innerHeight - skills.getBoundingClientRect().top;
    if(distancia_skills >= 300){
        let habilidades = document.getElementsByClassName("chart");
        // Asignamos las clases específicas definidas en el CSS o JS
        habilidades[0].classList.add("chart1"); // Python
        habilidades[1].classList.add("chart2"); // Airflow
        habilidades[2].classList.add("chart3"); // GCP
        habilidades[3].classList.add("chart4"); // SQL
        habilidades[4].classList.add("chart5"); // PowerBI
        habilidades[5].classList.add("chart6"); // Azure

        cargarAnimaciones();
    }
}

// ESTA ES LA PARTE CLAVE PARA TU PERFIL DE DATOS
function cargarAnimaciones(){
    $(function(){
        // 1. Python & Pandas (Azul Corporativo)
        $('.chart1').easyPieChart({
            size: 160,
            barColor: "#38bdf8", // Cyan Neon
            trackColor: "#1e293b", // Fondo oscuro
            scaleLength: 0,
            lineWidth: 10, // Un poco más fino es más elegante
            lineCap: "circle",
            animate: 2000,
        });
        // 2. Apache Airflow (Rojo marca Airflow)
        $('.chart2').easyPieChart({
            size: 160,
            barColor: "#38bdf8", // Cyan Neon
            trackColor: "#1e293b", // Fondo oscuro
            scaleLength: 0,
            lineWidth: 10, // Un poco más fino es más elegante
            lineCap: "circle",
            animate: 2000,
        });
        // 3. Google Cloud / BigQuery (Amarillo/Dorado)
        $('.chart3').easyPieChart({
            size: 160,
            barColor: "#38bdf8", // Cyan Neon
            trackColor: "#1e293b", // Fondo oscuro
            scaleLength: 0,
            lineWidth: 10, // Un poco más fino es más elegante
            lineCap: "circle",
            animate: 2000,
        });
        // 4. SQL / PL-SQL (Gris Claro)
        $('.chart4').easyPieChart({
            size: 160,
            barColor: "#38bdf8", // Cyan Neon
            trackColor: "#1e293b", // Fondo oscuro
            scaleLength: 0,
            lineWidth: 10, // Un poco más fino es más elegante
            lineCap: "circle",
            animate: 2000,
        });
        // 5. Power BI / Viz (Naranja)
        $('.chart5').easyPieChart({
            size: 160,
            barColor: "#38bdf8", // Cyan Neon
            trackColor: "#1e293b", // Fondo oscuro
            scaleLength: 0,
            lineWidth: 10, // Un poco más fino es más elegante
            lineCap: "circle",
            animate: 2000,
        });
        // 6. Azure & DevOps (Azul Claro)
        $('.chart6').easyPieChart({
            size: 160,
            barColor: "#38bdf8", // Cyan Neon
            trackColor: "#1e293b", // Fondo oscuro
            scaleLength: 0,
            lineWidth: 10, // Un poco más fino es más elegante
            lineCap: "circle",
            animate: 2000,
        });
    })
}

window.onscroll = function(){
    efectoHabilidades();
}