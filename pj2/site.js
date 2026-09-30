const progressBar = document.querySelector('.reading-progress span');
const siteHeader = document.querySelector('.site-header');
const hero = document.querySelector('.hero');

function updateProgress() {
  const scrollable = document.documentElement.scrollHeight - window.innerHeight;
  const progress = scrollable > 0 ? window.scrollY / scrollable : 0;
  progressBar.style.width = `${Math.min(1, Math.max(0, progress)) * 100}%`;
  const heroBoundary = hero.offsetTop + hero.offsetHeight - siteHeader.offsetHeight;
  siteHeader.classList.toggle('content-mode', window.scrollY >= heroBoundary);
}

updateProgress();
window.addEventListener('scroll', updateProgress, { passive: true });
window.addEventListener('resize', updateProgress);

window.addEventListener('load', () => {
  if (window.location.hash) {
    document.querySelector(window.location.hash)?.scrollIntoView();
  }
  updateProgress();
});

const profileTools = document.querySelector('.profile-tools');
const profileTrigger = profileTools.querySelector('.profile-trigger');
const profilePanel = profileTools.querySelector('.profile-panel');

function setProfileOpen(open) {
  profileTools.classList.toggle('open', open);
  profileTrigger.setAttribute('aria-expanded', String(open));
  profileTrigger.setAttribute('aria-label', open ? 'Close author profile' : 'Open author profile');
  profilePanel.setAttribute('aria-hidden', String(!open));
  profilePanel.inert = !open;
}

profileTrigger.addEventListener('click', () => setProfileOpen(!profileTools.classList.contains('open')));
document.addEventListener('click', (event) => {
  if (!profileTools.contains(event.target)) setProfileOpen(false);
});
window.addEventListener('keydown', (event) => {
  if (event.key === 'Escape') setProfileOpen(false);
});

document.querySelectorAll('[data-compare]').forEach((comparison) => {
  const slider = comparison.querySelector('input[type="range"]');
  const update = () => comparison.style.setProperty('--position', `${slider.value}%`);
  slider.addEventListener('input', update);
  update();
});

document.querySelectorAll('[data-distance]').forEach((control) => {
  const slider = control.querySelector('input[type="range"]');
  const update = () => {
    const progress = Number(slider.value) / Number(slider.max);
    const size = 86 - progress * 70;
    control.style.setProperty('--distance-size', `${size}%`);
    slider.setAttribute('aria-valuetext', `${Math.round(progress * 100)}% toward far view`);
  };
  slider.addEventListener('input', update);
  update();
});

const thresholdButtons = [...document.querySelectorAll('.threshold-switcher button')];
const thresholdImage = document.getElementById('thresholdImage');
const thresholdFormula = document.getElementById('thresholdFormula');
const thresholdChoice = document.getElementById('thresholdChoice');
const thresholdHeading = document.getElementById('thresholdHeading');
const thresholdCopy = document.getElementById('thresholdCopy');

thresholdButtons.forEach((button) => {
  button.addEventListener('click', () => {
    const { threshold, image, heading, copy } = button.dataset;
    thresholdButtons.forEach((item) => {
      const selected = item === button;
      item.classList.toggle('active', selected);
      item.setAttribute('aria-pressed', String(selected));
    });
    thresholdImage.src = image;
    thresholdImage.alt = `Finite-difference edge image at threshold ${threshold}`;
    thresholdFormula.textContent = `‖∇I‖ ≥ ${threshold}`;
    thresholdChoice.textContent = `${threshold === '0.20' ? 'CHOSEN THRESHOLD' : 'THRESHOLD PREVIEW'} · ${threshold}`;
    thresholdHeading.textContent = heading;
    thresholdCopy.textContent = copy;
  });
});

const tocLinks = [...document.querySelectorAll('.toc a')];
const sections = [...document.querySelectorAll('[data-section]')];
const tocObserver = new IntersectionObserver((entries) => {
  const visible = entries
    .filter((entry) => entry.isIntersecting)
    .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
  if (!visible) return;
  tocLinks.forEach((link) => link.classList.toggle('active', link.hash === `#${visible.target.id}`));
}, { rootMargin: '-20% 0px -60% 0px', threshold: [0, .1, .4] });
sections.forEach((section) => tocObserver.observe(section));

const lightbox = document.getElementById('lightbox');
const lightboxImage = lightbox.querySelector('img');
const lightboxTitle = lightbox.querySelector('h2');
const lightboxMeta = lightbox.querySelector('p');
const galleryCards = [...document.querySelectorAll('.gallery-card')];
let activeGalleryIndex = -1;

function openGalleryItem(index) {
  const card = galleryCards[index];
  if (!card) return;
  activeGalleryIndex = index;
  lightboxImage.src = card.dataset.full;
  lightboxImage.alt = card.querySelector('img').alt;
  lightboxTitle.textContent = card.dataset.title;
  lightboxMeta.textContent = card.dataset.meta;
  if (!lightbox.open) lightbox.showModal();
}

galleryCards.forEach((card, index) => {
  card.addEventListener('click', () => openGalleryItem(index));
});

lightbox.querySelector('.lightbox-close').addEventListener('click', () => lightbox.close());
lightbox.addEventListener('click', (event) => {
  if (event.target === lightbox) lightbox.close();
});
lightbox.addEventListener('close', () => {
  lightboxImage.src = '';
  activeGalleryIndex = -1;
});

window.addEventListener('keydown', (event) => {
  if (!lightbox.open || activeGalleryIndex < 0) return;
  if (event.key === 'ArrowRight') {
    openGalleryItem((activeGalleryIndex + 1) % galleryCards.length);
  } else if (event.key === 'ArrowLeft') {
    openGalleryItem((activeGalleryIndex - 1 + galleryCards.length) % galleryCards.length);
  }
});

function scientific(value) {
  if (value === 0) return '0';
  const [coefficient, exponent] = Number(value).toExponential(2).split('e');
  const superscripts = { '-': '⁻', '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹' };
  const formattedExponent = String(Number(exponent)).split('').map((character) => superscripts[character]).join('');
  return `${coefficient} × 10${formattedExponent}`;
}

fetch('web_assets/metrics.json')
  .then((response) => response.json())
  .then((metrics) => {
    const scientificKeys = new Set([
      'four_loop_same_error', 'two_loop_same_error', 'four_loop_full_error',
      'two_loop_full_error', 'dog_interior_mae', 'dog_interior_max_error',
      'unsharp_single_step_max_error',
      'apple_stack_reconstruction_error', 'orange_stack_reconstruction_error',
    ]);
    document.querySelectorAll('[data-metric]').forEach((element) => {
      const key = element.dataset.metric;
      if (scientificKeys.has(key)) {
        element.textContent = scientific(metrics[key]);
      } else if (key.endsWith('_ms')) {
        element.textContent = Number(metrics[key]).toFixed(2);
      } else if (key.includes('degrees')) {
        element.textContent = Number(metrics[key]).toFixed(3);
      } else if (key.endsWith('_mae')) {
        element.textContent = Number(metrics[key]).toFixed(4);
      } else if (key.endsWith('_percent')) {
        element.textContent = Number(metrics[key]).toFixed(2);
      }
    });
  })
  .catch(() => {
    // Static fallback values in the HTML keep the page readable from file://.
  });
