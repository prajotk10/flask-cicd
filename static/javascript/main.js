// Collapsibles
const collapsibles = document.querySelectorAll(".collapsible");

collapsibles.forEach((item) => {
  const iconContainer = item.querySelector(".toggler");

  if (iconContainer) {
    iconContainer.addEventListener("click", function (e) {
      e.stopPropagation();
      item.classList.toggle("collapsible--expanded");
    });
  }
});

// Drop-Down
document.addEventListener("DOMContentLoaded", function () {
  const dropdown = document.querySelector(".drop-down");
  const dropdownToggle = document.getElementById("dropdownToggle");

  dropdownToggle.addEventListener("click", function (e) {
    dropdown.classList.toggle("open");

    e.stopPropagation();
  });

  document.addEventListener("click", function (e) {
    if (!dropdown.contains(e.target)) {
      dropdown.classList.remove("open");
    }
  });
});

// Location
function showMap(event) {
  event.preventDefault();

  if (navigator.geolocation) {
    navigator.geolocation.getCurrentPosition(
      function (position) {
        const userLatitude = position.coords.latitude;
        const userLongitude = position.coords.longitude;

        const destination = "16.708005349161564,74.47823280612026";

        const directionsUrl = `https://www.google.com/maps/dir/?api=1&origin=${userLatitude},${userLongitude}&destination=${destination}&travelmode=driving`;

        window.location.href = directionsUrl;
      },

      function (error) {
        alert("Error getting location: " + error.message);
      }
    );
  } else {
    alert("Geolocation is not supported by this browser.");
  }
}

// Widget
const inputs = document.querySelectorAll(".tab__content input");
const imgWrapper = document.querySelector(".tab__img-wrapper");
const slides = document.querySelectorAll(".tab__img-wrapper img");
let currentIndex = 0;
let autoSlideInterval;

const firstSlide = slides[0].cloneNode(true);
const lastSlide = slides[slides.length - 1].cloneNode(true);
imgWrapper.appendChild(firstSlide);
imgWrapper.insertBefore(lastSlide, slides[0]);

imgWrapper.style.transform = `translateX(-100%)`;

function changeSlide(index) {
  imgWrapper.style.transition = "transform 0.5s ease";
  imgWrapper.style.transform = `translateX(-${(index + 1) * 100}%)`;
  currentIndex = index;
}

function startAutoSlide() {
  autoSlideInterval = setInterval(() => {
    currentIndex = (currentIndex + 1) % inputs.length;
    changeSlide(currentIndex);
    inputs[currentIndex].checked = true;
  }, 4000); // 4 seconds interval
}

imgWrapper.addEventListener("transitionend", () => {
  if (currentIndex === -1) {
    imgWrapper.style.transition = "none";
    imgWrapper.style.transform = `translateX(-${inputs.length * 100}%)`;
    currentIndex = inputs.length - 1;
  } else if (currentIndex === inputs.length) {
    imgWrapper.style.transition = "none";
    imgWrapper.style.transform = `translateX(-100%)`;
    currentIndex = 0;
  }
});

inputs.forEach((input, index) => {
  input.addEventListener("change", () => {
    changeSlide(index);
    clearInterval(autoSlideInterval);
    startAutoSlide();
  });
});

changeSlide(currentIndex);
startAutoSlide();

// Symptom Form
function validateSelection() {
  const checkboxes = document.querySelectorAll('input[type="checkbox"]');
  let selectedCount = 0;

  checkboxes.forEach((checkbox) => {
    if (checkbox.checked) {
      selectedCount++;
    }
  });

  const errorMessages = document.getElementsByClassName("error-message");
  const errorMessage = errorMessages[0];

  if (selectedCount < 3 || selectedCount > 5) {
    errorMessage.textContent = "Please! Select between 3 and 5 options.";
    errorMessage.classList.add("visible");
  } else {
    errorMessage.classList.remove("visible");
    alert("Form submitted successfully!");
  }
}

function clearForm() {
  document.querySelector("form").reset();
  const errorMessages = document.getElementsByClassName("error-message");
  for (let message of errorMessages) {
    message.classList.remove("visible");
  }
}

// AOS Animations
AOS.init();
