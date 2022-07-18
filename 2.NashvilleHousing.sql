

select *
from PortfolioProject.dbo.NashvilleHousing

-- Standarize Date Format

select SaleDate, convert(date, SaleDate) as SaleDateConverted
from PortfolioProject.dbo.NashvilleHousing

alter table NashvilleHousing
Add SaleDateConverted Date; 

update NashvilleHousing
set SaleDateConverted = CONVERT(DATE,SaleDate)


-- Populate property Address data 

Select *
from PortfolioProject.dbo.NashvilleHousing
where PropertyAddress is null 

Select *
from PortfolioProject.dbo.NashvilleHousing
order by ParcelID 


--There are many ParcelID that are the same but then some PropertyAddress are missing. If ParcelID =, PorpertyAddress = too

Select a.ParcelID, a.PropertyAddress, b.ParcelID, b.PropertyAddress, ISNULL(a.PropertyAddress, b.PropertyAddress)
from PortfolioProject.dbo.NashvilleHousing a
join PortfolioProject.dbo.NashvilleHousing b
	on a.parcelID = b.ParcelID
	AND a.[UniqueID ] <> b.[UniqueID ]
where a.PropertyAddress is null

UPDATE a
SET PropertyAddress = ISNULL(a.PropertyAddress, b.PropertyAddress)
from PortfolioProject.dbo.NashvilleHousing a
join PortfolioProject.dbo.NashvilleHousing b
	on a.parcelID = b.ParcelID
	AND a.[UniqueID ] <> b.[UniqueID ]
where a.PropertyAddress is null

Select a.ParcelID, a.PropertyAddress, b.ParcelID, b.PropertyAddress, ISNULL(a.PropertyAddress, b.PropertyAddress)
from PortfolioProject.dbo.NashvilleHousing a
join PortfolioProject.dbo.NashvilleHousing b
	on a.parcelID = b.ParcelID
	AND a.[UniqueID ] <> b.[UniqueID ]
where a.PropertyAddress is null --No rows shows.



--Breaking out Address into (Address, City, State) SUBSTRING

Select PropertyAddress
from PortfolioProject.dbo.NashvilleHousing

SELECT
SUBSTRING(PropertyAddress, 1, CHARINDEX(',', PropertyAddress) -1) as Address,  -- -1 gets the text without the comma
SUBSTRING(PropertyAddress,CHARINDEX(',', PropertyAddress) + 1 , LEN(PropertyAddress)) as Address 
from PortfolioProject.dbo.NashvilleHousing

alter table NashvilleHousing
Add PropertySplitCity Nvarchar(255);

update NashvilleHousing
set PropertySplitCity = SUBSTRING(PropertyAddress, 1, CHARINDEX(',', PropertyAddress) -1)

alter table NashvilleHousing
Add PropertySplitAddress Nvarchar(255);

update NashvilleHousing
set PropertySplitAddress = SUBSTRING(PropertyAddress,CHARINDEX(',', PropertyAddress) + 1 , LEN(PropertyAddress))

Select *
From PortfolioProject.dbo.NashvilleHousing


-- Better method PARSENAME

Select OwnerAddress
from PortfolioProject.dbo.NashvilleHousing

select 
PARSENAME(REPLACE(OwnerAddress,',','.'), 3) -- Coge los trozos 'al reves' de como lo pensamos
,PARSENAME(REPLACE(OwnerAddress,',','.'), 2)
,PARSENAME(REPLACE(OwnerAddress,',','.'), 1) -- Este es el último trozo
from PortfolioProject.dbo.NashvilleHousing

alter table NashvilleHousing
Add OwnerSplitAddress Nvarchar(255);

update NashvilleHousing
set OwnerSplitAddress = PARSENAME(REPLACE(OwnerAddress,',','.'), 3)

alter table NashvilleHousing
Add OwnerSplitCity Nvarchar(255);

update NashvilleHousing
set OwnerSplitCity = PARSENAME(REPLACE(OwnerAddress,',','.'), 2)

alter table NashvilleHousing
Add OwnerSplitState Nvarchar(255);

update NashvilleHousing
set OwnerSplitState = PARSENAME(REPLACE(OwnerAddress,',','.'), 1)


-- Change Y and N to Yes and No in 'Sold as Vacant' fiend 

Select Distinct(SoldAsVacant), Count(SoldAsVacant) as Count 
From PortfolioProject.dbo.NashvilleHousing
Group by SoldAsVacant
order by 2

Select SoldAsVacant
, CASE When SoldAsVacant = 'Y' THEN 'Yes'
	   When SoldAsVacant = 'N' THEN 'No'
	   Else SoldAsVacant
	   End
From PortfolioProject.dbo.NashvilleHousing

update NashvilleHousing
set SoldAsVacant = CASE When SoldAsVacant = 'Y' THEN 'Yes'
	   When SoldAsVacant = 'N' THEN 'No'
	   Else SoldAsVacant
	   End

-- Remove Duplicates using CTE
WITH RowNumCTE AS(
Select *,
	ROW_NUMBER() OVER (
	PARTITION BY ParcelID,
	             PropertyAddress,
				 SalePrice,
				 LegalReference
				 ORDER BY 
					UniqueID
					) row_num

From PortfolioProject.dbo.NashvilleHousing
)
DELETE --Select * if you want to see data instead of deleting it first
From RowNumCTE 
where row_num > 1


-- Delete Unused Columns 

Select *
From PortfolioProject.dbo.NashvilleHousing

ALTER TABLE PortfolioProject.dbo.NashvilleHousing
DROP COLUMN OwnerAddress, TaxDistrict, PropertyAddress, SaleDate