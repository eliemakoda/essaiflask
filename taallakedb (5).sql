-- phpMyAdmin SQL Dump
-- version 5.2.1
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1
-- Generation Time: May 30, 2024 at 12:04 PM
-- Server version: 10.4.32-MariaDB
-- PHP Version: 8.2.12

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
-- Table structure for table `activitytb`
--

CREATE TABLE `activitytb` (
  `id` int(11) NOT NULL,
  `title` varchar(255) NOT NULL,
  `description` text NOT NULL,
  `status` varchar(20) NOT NULL,
  `activity_image` varchar(255) DEFAULT NULL,
  `date` date NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `activitytb`
--

INSERT INTO `activitytb` (`id`, `title`, `description`, `status`, `activity_image`, `date`) VALUES
(2, 'BFAR 4A at LGU ng Lipa City', 'BFAR 4A at LGU ng Lipa City, Nagsagawa ng Lake Seeding sa Lawa ng Taal', 'Activate', 'static/activity\\1.jpg', '2024-03-02'),
(3, ' Site Validation', 'BFAR 4A, Nagsagawa ng site validation at water quality assessment sa anim na ilog ng GEA, Cavite', 'Activate', 'static/activity\\2.jpg', '2024-02-28'),
(4, 'Fisheries Enhancement AR4A', 'Fisheries Enhancement sa ilog ng Pagsanjan, isinagawa ng BFAR4A', 'Activate', 'static/activity\\4.jpg', '2024-03-22'),
(5, 'BFAR 4A', 'BFAR 4A, Nagsagawa ng Capacity Building Training sa mga Mangingisda ng Ilog ng Pagsanjan', 'Activate', 'static/activity\\4.jpg', '2024-03-20'),
(6, 'BFAR 4A', 'BFAR 4A, Dumalo sa Mid-Year Accomplishment Review ng Basil', 'Activate', 'static/activity\\5.jpg', '2024-03-09'),
(7, 'BFAR 4A', 'BFAR 4A, nakiisa sa MOA signing at MetBuoy Development sa Lawa ng Taal', 'Activate', 'static/activity\\6.jpg', '2024-03-27');

-- --------------------------------------------------------

--
-- Table structure for table `announcementtb`
--

CREATE TABLE `announcementtb` (
  `id` int(11) NOT NULL,
  `title` varchar(255) NOT NULL,
  `description` text NOT NULL,
  `status` varchar(20) NOT NULL,
  `announcement_image` varchar(255) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `announcementtb`
--

INSERT INTO `announcementtb` (`id`, `title`, `description`, `status`, `announcement_image`) VALUES
(1, 'Taal', ' ', 'Activate', 'static/activity\\taalmap.png'),
(2, 'Taal', ' ', 'Activate', 'static/activity\\taal-volcano-eruption.png');

-- --------------------------------------------------------

--
-- Table structure for table `contacttb`
--

CREATE TABLE `contacttb` (
  `id` int(255) NOT NULL,
  `bldg_no` varchar(255) NOT NULL,
  `brgy` varchar(255) NOT NULL,
  `municipality` varchar(255) NOT NULL,
  `province` varchar(255) NOT NULL,
  `zip_code` varchar(255) NOT NULL,
  `mobile` varchar(255) NOT NULL,
  `email` varchar(255) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `contacttb`
--

INSERT INTO `contacttb` (`id`, `bldg_no`, `brgy`, `municipality`, `province`, `zip_code`, `mobile`, `email`) VALUES
(1, '', 'Ambulong', 'Tanauan City', 'Batangas', '4232', '+639208820144', 'bfarabitos@gmail.com');

-- --------------------------------------------------------

--
-- Table structure for table `servicestb`
--

CREATE TABLE `servicestb` (
  `id` int(11) NOT NULL,
  `title` varchar(255) NOT NULL,
  `description` text NOT NULL,
  `status` varchar(20) NOT NULL,
  `service_image` varchar(255) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `servicestb`
--

INSERT INTO `servicestb` (`id`, `title`, `description`, `status`, `service_image`) VALUES
(1, 'Fisheries Quarantine and Inspection', ' ', 'Activate', 'static/activity\\taallake.jpg'),
(2, 'Fishpond Lease Agreement', ' ', 'Activate', 'static/activity\\taallake.jpg'),
(3, 'Commercial Fishing Vessel and Gear Licensing', ' ', 'active', 'static/activity\\taallake.jpg'),
(4, 'Fisheries Inspection', ' ', 'active', 'static/activity\\taallake.jpg'),
(5, 'Regional Fisheries Laboratory', ' ', 'active', 'static/activity\\taallake.jpg'),
(6, 'Fingerlings Distribution', ' ', 'active', 'static/activity\\taallake.jpg');

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
  `time` varchar(255) NOT NULL,
  `weater-condition` varchar(255) NOT NULL,
  `wind-direction` varchar(255) NOT NULL,
  `color-of-water` varchar(255) NOT NULL,
  `air-temperature` varchar(255) NOT NULL,
  `water-transparency` varchar(255) NOT NULL,
  `water-temperature` varchar(255) NOT NULL,
  `Type` varchar(255) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `taaltb`
--

INSERT INTO `taaltb` (`id`, `stationid`, `Barangay`, `Month`, `Year`, `pH`, `Ammonia`, `DO`, `Nitrate`, `Phosphate`, `time`, `weater-condition`, `wind-direction`, `color-of-water`, `air-temperature`, `water-transparency`, `water-temperature`, `Type`) VALUES
(11, '4', 'Manalaw', 'March', '2024', '2', '88', '8', '8', '8', '23:15', 'sunny', 'SW', 'GREEN', '2', '2', '2', ''),
(12, '3', 'Nangkaan', 'March', '2024', '2', '2', '2', '2', '2', '23:15', 'sunny', 'SW', 'GREEN', '2', '2', '2', '');

-- --------------------------------------------------------

--
-- Table structure for table `usertb`
--

CREATE TABLE `usertb` (
  `id` int(255) NOT NULL,
  `fname` varchar(255) NOT NULL,
  `mname` varchar(255) NOT NULL,
  `lname` varchar(255) NOT NULL,
  `email` varchar(255) NOT NULL,
  `username` varchar(255) NOT NULL,
  `password` varchar(255) NOT NULL,
  `userType` varchar(255) NOT NULL,
  `status` varchar(255) NOT NULL,
  `profile_image` varchar(255) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `usertb`
--

INSERT INTO `usertb` (`id`, `fname`, `mname`, `lname`, `email`, `username`, `password`, `userType`, `status`, `profile_image`) VALUES
(1, '', '', '', '', 'admin', '$2a$12$4aW8XD84BtnI.DQvIJWVAual7MZWvKNGW.ojBiydVWlYsqBSQBh7a', 'admin', '', 'static/activity\\IMG_5624.JPG'),
(2, 'asdf', 'asdf', 'asdf', 'marcluigitimola@gmail.com', 'asdf', '$2b$12$D4LadsU/bh7qGqtkhE2eMelndpN/wZ0iTj/z6sMh0sWYTAtkPds7e', 'Staff', 'active', 'static/activity\\Screenshot_2024-01-30_220335.png'),
(3, 'test', 'test', 'test', 'test', 'test', '$2b$12$VAZo8cfKPg573PQ4cPfXaegXlMNXjyQoCJWdu71hGWEYBJ9EclwKy', 'Staff', 'active', 'static/activity\\370319118_312698827930984_2208904944267846325_n.png');

--
-- Indexes for dumped tables
--

--
-- Indexes for table `activitytb`
--
ALTER TABLE `activitytb`
  ADD PRIMARY KEY (`id`);

--
-- Indexes for table `announcementtb`
--
ALTER TABLE `announcementtb`
  ADD PRIMARY KEY (`id`);

--
-- Indexes for table `contacttb`
--
ALTER TABLE `contacttb`
  ADD PRIMARY KEY (`id`);

--
-- Indexes for table `servicestb`
--
ALTER TABLE `servicestb`
  ADD PRIMARY KEY (`id`);

--
-- Indexes for table `taaltb`
--
ALTER TABLE `taaltb`
  ADD PRIMARY KEY (`id`);

--
-- Indexes for table `usertb`
--
ALTER TABLE `usertb`
  ADD PRIMARY KEY (`id`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `activitytb`
--
ALTER TABLE `activitytb`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=8;

--
-- AUTO_INCREMENT for table `announcementtb`
--
ALTER TABLE `announcementtb`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=3;

--
-- AUTO_INCREMENT for table `contacttb`
--
ALTER TABLE `contacttb`
  MODIFY `id` int(255) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=2;

--
-- AUTO_INCREMENT for table `servicestb`
--
ALTER TABLE `servicestb`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=7;

--
-- AUTO_INCREMENT for table `taaltb`
--
ALTER TABLE `taaltb`
  MODIFY `id` int(255) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=13;

--
-- AUTO_INCREMENT for table `usertb`
--
ALTER TABLE `usertb`
  MODIFY `id` int(255) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=4;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
