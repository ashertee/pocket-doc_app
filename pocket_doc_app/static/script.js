// Add function to capture symptoms
function addSymptom() {
    const symptomInput = document.getElementById('symptom-input');
    const symptomList = document.getElementById('symptom-list');
    const listItem = document.createElement('li');
    listItem.textContent = symptomInput.value;
    // build a list of input symptom
    symptomList.appendChild(listItem);
    symptomInput.value = '';
}

// Post & fetch the diagnosis
document.getElementById('symptom-form').onsubmit = function (event) {
    event.preventDefault();
    const symptoms = [];
    document.querySelectorAll('#symptom-list li').forEach(item => {
        symptoms.push(item.textContent);
    });
    fetch('/diagnose', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({symptoms: symptoms})
    })
        .then(response => response.json())
        .then(data => {
            document.getElementById('diagnosis-result').textContent = 'Diagnosis: ' + data.diagnosis;
        });
};


// Add some dynamic aspect
document.getElementById('loadMore').addEventListener('click', function () {
    fetch('/more-content')
        .then(response => response.json())
        .then(data => {
            const content = document.getElementById('content');
            data.forEach(item => {
                const div = document.createElement('div');
                div.classList.add('item');
                div.innerHTML = item.html;
                content.appendChild(div);
            });
        });
});


// toggle dark mode
const toggleSwitch = document.querySelector('.theme-switch input[type="checkbox"]');

toggleSwitch.addEventListener('change', function () {
    document.body.classList.toggle('dark-mode');
});



