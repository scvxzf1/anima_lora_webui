/* Scroll-triggered animation observer.
 * Adds .dragon-in-view to elements with .dragon-reveal when they enter viewport.
 * Also handles parallax effect for .dragon-parallax elements.
 */

let revealObserver = null;

export function initScrollAnimations() {
    if (revealObserver) revealObserver.disconnect();

    revealObserver = new IntersectionObserver(
        (entries) => {
            for (const entry of entries) {
                if (entry.isIntersecting) {
                    entry.target.classList.add('dragon-in-view');
                    revealObserver.unobserve(entry.target);
                }
            }
        },
        { threshold: 0.15, rootMargin: '0px 0px -60px 0px' }
    );

    scanForReveal();
}

export function scanForReveal() {
    if (!revealObserver) return;
    document.querySelectorAll('.dragon-reveal:not(.dragon-in-view)').forEach((el) => {
        revealObserver.observe(el);
    });
}

let parallaxTicking = false;
function handleParallax() {
    if (parallaxTicking) return;
    parallaxTicking = true;
    requestAnimationFrame(() => {
        const scrolled = window.scrollY;
        document.querySelectorAll('.dragon-parallax').forEach((el) => {
            const speed = parseFloat(el.dataset.parallaxSpeed || '0.3');
            el.style.transform = `translateY(${scrolled * speed * -0.1}px)`;
        });
        parallaxTicking = false;
    });
}

export function initParallax() {
    window.addEventListener('scroll', handleParallax, { passive: true });
}
