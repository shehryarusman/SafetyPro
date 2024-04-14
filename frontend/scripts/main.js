$(function(){
    $('#header').load("includes/header.html", function() {
    });

    const element = document.getElementById('video');
    const windowHeight = window.innerHeight;
    element.style.height = `${windowHeight}px`;
});

window.addEventListener('scroll', function() {
    var navbar = document.querySelector('#header');
    if (window.scrollY > 50) {
        navbar.classList.add('scrolled');
    } else {
        navbar.classList.remove('scrolled');
    }
});

window.addEventListener('scroll', function() {
    const elements = document.querySelectorAll('.animate-on-scroll');
    elements.forEach((element) => {
        const elementTop = element.getBoundingClientRect().top;
        const windowHeight = window.innerHeight;
        if (elementTop < windowHeight * 0.75) {
          element.classList.add('animate');
        } else {
            if (elementTop > windowHeight){
                element.classList.remove('animate');
            }
        }
      });
    }
);

$(function(){
    const elements = document.querySelectorAll('.animate-on-scroll');
    elements.forEach((element) => {
        //const elementTop = element.getBoundingClientRect().top;
        //element.classList.add('animate');
    });
});

window.addEventListener('resize', function() {
    const element = document.getElementById('video');
    const windowHeight = window.innerHeight;
    element.style.height = `${windowHeight}px`;
});