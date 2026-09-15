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

// Images determine much of the article's final height. Revisit a deep link once
// they have loaded so direct links land on the requested section without shift.
window.addEventListener('load', () => {
  if (!window.location.hash) return;
  const target = document.querySelector(window.location.hash);
  if (target) {
    target.scrollIntoView();
    updateProgress();
  }
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

profileTrigger.addEventListener('click', () => {
  setProfileOpen(!profileTools.classList.contains('open'));
});
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

const metricButtons = [...document.querySelectorAll('.metric-tabs button')];
const metricImage = document.getElementById('metricHeatmap');
const metricName = document.getElementById('metricName');
const metricCopy = document.getElementById('metricCopy');

metricButtons.forEach((button) => {
  button.addEventListener('click', () => {
    metricButtons.forEach((item) => {
      const selected = item === button;
      item.classList.toggle('active', selected);
      item.setAttribute('aria-selected', String(selected));
    });
    metricImage.src = button.dataset.image;
    metricImage.alt = `${button.dataset.name} score heatmap over a 31 by 31 displacement window`;
    metricName.textContent = button.dataset.name;
    metricCopy.textContent = button.dataset.copy;
  });
});

const tocLinks = [...document.querySelectorAll('.toc a')];
const sections = [...document.querySelectorAll('[data-section]')];
const tocObserver = new IntersectionObserver((entries) => {
  const visible = entries
    .filter((entry) => entry.isIntersecting)
    .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
  if (!visible) return;
  tocLinks.forEach((link) => {
    link.classList.toggle('active', link.hash === `#${visible.target.id}`);
  });
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
