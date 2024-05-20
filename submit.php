<?php
// Establish database connection
$servername = "localhost";
$username = "root";
$password = "";
$dbname = "leave_management";

$conn = mysqli_connect($servername, $username, $password, $dbname);

// Check connection
if (!$conn) {
    die("Connection failed: " . mysqli_connect_error());
}

if ($_SERVER["REQUEST_METHOD"] == "POST") {
    // Retrieve form data
    $name = $_POST['name'];
    $startDate = $_POST['startDate'];
    $endDate = $_POST['endDate'];
    $reason = $_POST['reason'];

    // Debug: Output form data to confirm it's being received
    //echo "Name: " . $name . "<br>";
    //echo "Start Date: " . $startDate . "<br>";
    //echo "End Date: " . $endDate . "<br>";
    //echo "Leave Reason: " . $reason . "<br>";

    // File upload handling
    $documentFileName = $_FILES['document']['name'];
    $documentTmpName = $_FILES['document']['tmp_name'];

    // Move uploaded file to a folder (optional)
    $uploadDirectory = "uploads/";
    $documentFilePath = $uploadDirectory . $documentFileName;
    move_uploaded_file($documentTmpName, $documentFilePath);

    $sql = "INSERT INTO `leaves` (name, start_date, end_date, reason, document) VALUES ('$name', '$startDate', '$endDate', '$reason', '$documentFilePath')";

    if ($conn->query($sql) === TRUE) {
        header("Location: submit.php");
        exit();
    } else {
        echo "Error: " . $sql . "<br>" . $conn->error;
    }
}

$conn->close();
//exit;
?>