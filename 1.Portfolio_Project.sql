
select *
from PortfolioProject..CovidDeaths
order by 3,4

--select *
--from PortfolioProject..CovidVaccinations
--order by 3,4

--Seleccionar datos 
Select Location, date, total_cases, new_cases, total_deaths, population 
From PortfolioProject..CovidDeaths
order by 1,2

--Buscar Casos Totales vs Muertes Totales en España
Select Location, date, total_cases, total_deaths, (total_deaths/total_cases)*100 as DeathPercentage
From PortfolioProject..CovidDeaths
Where location like 'Spain%'
order by 1,2

--Buscar Casos Totales vs Población en España
Select Location, date, total_cases, population,  (total_cases/population)*100 as CovidPercentage
From PortfolioProject..CovidDeaths
Where location like 'Spain%'
order by 1,2


--Comparar el porcentaje de infección en diferentes paises 

--Buscar Casos Totales vs Población en España
Select Location, population, MAX(total_cases) as HighestInfecctionCount,  MAX(total_cases/population)*100 as CovidPercentage
From PortfolioProject..CovidDeaths
Group by Location, population
order by CovidPercentage desc

--Mostrar los paises con mas muertes
--Total_deaths estaba en varchar, lo pasamos a numérico
--Cuando el contintente = null, location = continent y no nos interesa
Select Location, MAX(cast(Total_deaths as int)) as TotalDeathCount
From PortfolioProject..CovidDeaths
where continent is not null
Group by Location
order by TotalDeathCount desc


-- Mostrar diferencias por contintentes 
Select continent, MAX(cast(Total_deaths as int)) as TotalDeathCount
From PortfolioProject..CovidDeaths
where continent is not null -- Hay continentes = NULL 
Group by continent
order by TotalDeathCount desc

--Infecciones globales por dia 

Select date, SUM(new_cases) as GlobalCases, SUM(cast(new_deaths as int)) as GlobalDeaths
From PortfolioProject..CovidDeaths
Group by date
order by 1


--Juntar las dos bases de datos 
Select *
from PortfolioProject..CovidDeaths dea
Join PortfolioProject..CovidVaccinations vac 
	On dea.location = vac.location
	and dea.date = vac.date



Select dea.continent, dea.location, dea.date, dea.population, cast(vac.new_vaccinations as int) as New_vaccinations, cast(dea.new_deaths as int) as new_deaths --Importante especificar de qué base de datos seleccionamos (no solo date)
from PortfolioProject..CovidDeaths dea
Join PortfolioProject..CovidVaccinations vac 
	On dea.location = vac.location
	and dea.date = vac.date
where dea.continent is not null
order by new_deaths desc

--Realizar conteo de vacunaciones por país y día
Select dea.continent, dea.location, dea.date, dea.population, cast(vac.new_vaccinations as int) as new_vaccinations, 
SUM(CONVERT(int,new_vaccinations)) OVER (Partition by dea.Location Order by dea.location, dea.Date) as TotalCountryVaccinations --Contar nuevos casos, acabar la cuenta por País
from PortfolioProject..CovidDeaths dea
Join PortfolioProject..CovidVaccinations vac 
	On dea.location = vac.location
	and dea.date = vac.date
where dea.continent is not null
order by 2,3

--Usar CTE
With PopvsVac (Continent, Location, Date, Population, new_vaccinations, TotalCountryVaccinations)
as 
(
Select dea.continent, dea.location, dea.date, dea.population, cast(vac.new_vaccinations as int) as new_vaccinations, 
SUM(CONVERT(int,new_vaccinations)) OVER (Partition by dea.Location Order by dea.location, dea.Date) as TotalCountryVaccinations --Contar nuevos casos, acabar la cuenta por País
from PortfolioProject..CovidDeaths dea
Join PortfolioProject..CovidVaccinations vac 
	On dea.location = vac.location
	and dea.date = vac.date
where dea.continent is not null
)
Select *, (TotalCountryVaccinations/Population	)*100 as TotalCountryVaccinationsPercentaje
From PopvsVac



--TEMP TABLE 
Drop table if exists #PercentPopulationVaccinated --Esta función elimina la tabla si ya existe, en caso de querer actualizarla
Create Table #PercentPopulationVaccinated
(
Continent nvarchar(255),
Location nvarchar(255),
Date datetime,
Population numeric,
New_vaccinations numeric,
TotalCountryVaccinations numeric,
)


Insert into #PercentPopulationVaccinated
Select dea.continent, dea.location, dea.date, dea.population, cast(vac.new_vaccinations as int) as new_vaccinations, 
SUM(CONVERT(int,new_vaccinations)) OVER (Partition by dea.Location Order by dea.location, dea.Date) as TotalCountryVaccinations --Contar nuevos casos, acabar la cuenta por País
from PortfolioProject..CovidDeaths dea
Join PortfolioProject..CovidVaccinations vac 
	On dea.location = vac.location
	and dea.date = vac.date
where dea.continent is not null

Select * 
From #PercentPopulationVaccinated