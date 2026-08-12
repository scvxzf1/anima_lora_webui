/* Scroll-triggered animation observer.
 * Adds .apple-in-view to elements with .apple-reveal when they enter viewport.
 * Also handles parallax effect for .apple-parallax elements.
 */

let revealObserver = null;

export function initScrollAnimations() {
    if (revealObserver) revealObserver.disconnect();

    revealObserver = new IntersectionObserver(
        (entries) => {
            for (const entry of entries) {
                if (entry.isIntersecting) {
                    entry.target.classList.add('apple-in-view');
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
    document.querySelectorAll('.apple-reveal:not(.apple-in-view)').forEach((el) => {
        revealObserver.observe(el);
    });
}

let parallaxTicking = false;
function handleParallax() {
    if (parallaxTicking) return;
    parallaxTicking = true;
    requestAnimationFrame(() => {
        const scrolled = window.scrollY;
        document.querySelectorAll('.apple-parallax').forEach((el) => {
            const speed = parseFloat(el.dataset.parallaxSpeed || '0.3');
            el.style.transform = `translateY(${scrolled * speed * -0.1}px)`;
        });
        parallaxTicking = false;
    });
}

export function initParallax() {
    window.addEventListener('scroll', handleParallax, { passive: true });
}
