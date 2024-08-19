function addSymptom() {
    const symptomInput = document.getElementById('symptom-input');
    const symptomList = document.getElementById('symptom-list');
    const listItem = document.createElement('li');
    listItem.textContent = symptomInput.value;
    symptomList.appendChild(listItem);
    symptomInput.value = '';
}

document.getElementById('symptom-form').onsubmit = function(event) {
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
        body: JSON.stringify({ symptoms: symptoms })
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('diagnosis-result').textContent = 'Diagnosis: ' + data.diagnosis;
    });
};


// Add some dynamic aspect
document.getElementById('loadMore').addEventListener('click', function() {
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


// Example toggle dark mode
const toggleSwitch = document.querySelector('.theme-switch input[type="checkbox"]');

toggleSwitch.addEventListener('change', function() {
  document.body.classList.toggle('dark-mode');
});

        //
        //
        // // Modal functionality
        // var modal = document.getElementById("diagnosisModal");
        // var span = document.getElementsByClassName("close")[0];
        //
        // function openModal(diagnosis) {
        //     var title = document.getElementById("modal-title");
        //     var description = document.getElementById("modal-description");
        //
        //     title.textContent = diagnosis;
        //     description.textContent = diagnosisDescriptions[diagnosis] || "No description available for this diagnosis.";
        //
        //     modal.style.display = "display";
        // }
        //
        // span.onclick = function() {
        //     modal.style.display = "none";
        // }
        //
        // window.onclick = function(event) {
        //     if (event.target == modal) {
        //         modal.style.display = "none";
        //     }
        // }


        //
//
//    function removeSymptom(index) {
//             selectedSymptoms.splice(index, 1);
//             displaySymptoms();
//         }
//
//         function displaySymptoms() {
//             var symptomsList = document.getElementById('symptoms-list');
//             symptomsList.innerHTML = '';
//             selectedSymptoms.forEach(function(symptom, index) {
//                 var li = document.createElement('li');
//                 li.textContent = symptom;
//                 var removeBtn = document.createElement('button');
//                 removeBtn.textContent = 'Remove';
//                 removeBtn.className = 'remove-btn';
//                 removeBtn.onclick = function() { removeSymptom(index); };
//                 li.appendChild(removeBtn);
//                 symptomsList.appendChild(li);
//             });
//         }



        //    // Fetch the description from the server
        // function fetchDescription(diagnosis, element) {
        //     fetch('/get_description/' + encodeURIComponent(diagnosis))
        //         .then(response => response.json())
        //         .then(data => {
        //             var descriptionContainer = document.createElement('div');
        //             descriptionContainer.className = 'description-container';
        //             var descriptionText = document.createElement('p');
        //             descriptionText.textContent = data.description;
        //             descriptionContainer.appendChild(descriptionText);
        //
        //             // Toggle visibility
        //             if (element.nextElementSibling && element.nextElementSibling.className === 'description-container') {
        //                 element.nextElementSibling.remove();  // Remove existing description if present
        //             } else {
        //                 element.parentNode.insertBefore(descriptionContainer, element.nextSibling);
        //             }
        //         });
        // }


