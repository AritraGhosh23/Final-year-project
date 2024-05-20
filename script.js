document.getElementById('leaveForm').addEventListener('submit', function (event) {
    var startDate = new Date(document.getElementById('startDate').value);
    var endDate = new Date(document.getElementById('endDate').value);

    if (startDate > endDate) {
        alert('End date cannot be before start date');
        event.preventDefault();
    }
});
