-- phpMyAdmin SQL Dump
-- version 5.2.2-dev+20230521.442a30e226
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1
-- Generation Time: Mar 23, 2024 at 08:31 AM
-- Server version: 10.4.24-MariaDB
-- PHP Version: 8.1.4

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `taallakedb`
--

-- --------------------------------------------------------

--
-- Table structure for table `taaltb`
--

CREATE TABLE `taaltb` (
  `id` int(255) NOT NULL,
  `stationid` varchar(255) NOT NULL,
  `Barangay` varchar(255) NOT NULL,
  `Month` varchar(255) NOT NULL,
  `Year` varchar(255) NOT NULL,
  `pH` varchar(255) NOT NULL,
  `Ammonia` varchar(255) NOT NULL,
  `DO` varchar(255) NOT NULL,
  `Nitrate` varchar(255) NOT NULL,
  `Phosphate` varchar(255) NOT NULL,
  `Type` varchar(255) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Dumping data for table `taaltb`
--

INSERT INTO `taaltb` (`id`, `stationid`, `Barangay`, `Month`, `Year`, `pH`, `Ammonia`, `DO`, `Nitrate`, `Phosphate`, `Type`) VALUES
(1, '1', 'Manalaw', 'January', '2023', '1', '1', '1', '1', '1', 'Lake'),
(2, '1', 'Manalaw', 'January', '2023', '1', '1', '1', '1', '1', 'Lake'),
(3, '2', 'Manalaw', 'February', '2023', '2', '2', '2', '2', '2', 'River'),
(4, '3', 'Nangkaan', 'March', '2023', '2.0', '2.0', '2.0', '2.0', '1.0', 'River'),
(5, '1', 'Buso Buso', 'January', '2023', '8.4', '0.087', '0.001', '0.343', '1.0', 'Lake'),
(6, '1', 'Stn. I (Central West bay)', 'February', '2023', '2', '2', '2', '2', '2023', 'Lake'),
(7, '1', 'Stn. II (East Bay)', 'November', '2023', '1', '1', '1', '1', '2023', 'Lake'),
(8, '2', 'Nangkaan', 'March', '2024', '1.0', '1.0', '1.0', '1.0', '1.0', ''),
(9, '2', 'Nangkaan', 'February', '2024', '1.0', '1.0', '1.0', '1.0', '1.0', ''),
(10, '2', 'Bañaga', 'February', '2024', '1.0', '1.0', '1.0', '1.0', '1.0', ''),
(11, '4', 'Manalaw', 'March', '2024', '8.0', '88.0', '8.0', '8.0', '8.0', '');

--
-- Indexes for dumped tables
--

--
-- Indexes for table `taaltb`
--
ALTER TABLE `taaltb`
  ADD PRIMARY KEY (`id`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `taaltb`
--
ALTER TABLE `taaltb`
  MODIFY `id` int(255) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=12;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
