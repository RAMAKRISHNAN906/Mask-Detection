/* ═══════════════════════════════════════════════════════════════
   MaskGuard AI - GSAP Animations Module
   ═══════════════════════════════════════════════════════════════ */

document.addEventListener('DOMContentLoaded', () => {

    // Register GSAP ScrollTrigger
    if (typeof gsap !== 'undefined' && typeof ScrollTrigger !== 'undefined') {
        gsap.registerPlugin(ScrollTrigger);
    }

    // ── Navbar Entrance ───────────────────────────────────────
    gsap.from('.navbar', {
        y: -80,
        opacity: 0,
        duration: 0.8,
        delay: 1.8,
        ease: 'power3.out'
    });

    // ── Hero Animations ───────────────────────────────────────
    const heroContent = document.querySelector('.hero-content');
    if (heroContent) {
        const heroTl = gsap.timeline({ delay: 2 });

        heroTl
            .from('.hero-badge', {
                y: 30, opacity: 0, duration: 0.6, ease: 'power3.out'
            })
            .from('.hero-title', {
                y: 40, opacity: 0, duration: 0.7, ease: 'power3.out'
            }, '-=0.3')
            .from('.hero-subtitle', {
                y: 30, opacity: 0, duration: 0.6, ease: 'power3.out'
            }, '-=0.3')
            .from('.hero-actions .btn', {
                y: 20, opacity: 0, duration: 0.5, stagger: 0.15, ease: 'power3.out'
            }, '-=0.2')
            .from('.hero-stat', {
                y: 20, opacity: 0, duration: 0.4, stagger: 0.1, ease: 'power3.out'
            }, '-=0.2')
            .from('.hero-visual', {
                x: 60, opacity: 0, duration: 0.8, ease: 'power3.out'
            }, '-=0.6');
    }

    // ── Feature Cards Hover Effects ───────────────────────────
    document.querySelectorAll('.feature-card').forEach(card => {
        card.addEventListener('mouseenter', () => {
            gsap.to(card, {
                y: -8,
                duration: 0.3,
                ease: 'power2.out'
            });
        });

        card.addEventListener('mouseleave', () => {
            gsap.to(card, {
                y: 0,
                duration: 0.3,
                ease: 'power2.out'
            });
        });
    });

    // ── Glass Card Hover Tilt ─────────────────────────────────
    document.querySelectorAll('.glass-card').forEach(card => {
        card.addEventListener('mousemove', (e) => {
            const rect = card.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            const centerX = rect.width / 2;
            const centerY = rect.height / 2;

            const rotateX = ((y - centerY) / centerY) * -2;
            const rotateY = ((x - centerX) / centerX) * 2;

            card.style.transform = `perspective(1000px) rotateX(${rotateX}deg) rotateY(${rotateY}deg)`;
        });

        card.addEventListener('mouseleave', () => {
            card.style.transform = '';
        });
    });

    // ── Stat Counter Animations ───────────────────────────────
    document.querySelectorAll('.dash-stat-value').forEach(el => {
        const text = el.textContent.trim();
        const isPercentage = text.includes('%');
        const numStr = text.replace('%', '');
        const num = parseFloat(numStr);

        if (isNaN(num)) return;

        ScrollTrigger.create({
            trigger: el,
            start: 'top 80%',
            once: true,
            onEnter: () => {
                gsap.from(el, {
                    textContent: 0,
                    duration: 1.5,
                    ease: 'power2.out',
                    snap: { textContent: num % 1 !== 0 ? 0.1 : 1 },
                    onUpdate: function () {
                        const current = parseFloat(el.textContent);
                        el.textContent = (num % 1 !== 0 ? current.toFixed(1) : Math.round(current)) +
                            (isPercentage ? '%' : '');
                    },
                    onComplete: () => {
                        el.textContent = text;
                    }
                });
            }
        });
    });

    // ── Smooth Page Transitions ───────────────────────────────
    gsap.from('.main-content', {
        opacity: 0,
        y: 20,
        duration: 0.5,
        delay: 1.9,
        ease: 'power2.out'
    });

    // ── Button Ripple Effect ──────────────────────────────────
    document.querySelectorAll('.btn').forEach(btn => {
        btn.addEventListener('click', function (e) {
            const ripple = document.createElement('span');
            const rect = this.getBoundingClientRect();
            const size = Math.max(rect.width, rect.height);
            const x = e.clientX - rect.left - size / 2;
            const y = e.clientY - rect.top - size / 2;

            ripple.style.cssText = `
                position: absolute;
                width: ${size}px;
                height: ${size}px;
                left: ${x}px;
                top: ${y}px;
                background: rgba(255,255,255,0.2);
                border-radius: 50%;
                pointer-events: none;
                transform: scale(0);
            `;

            this.style.position = 'relative';
            this.style.overflow = 'hidden';
            this.appendChild(ripple);

            gsap.to(ripple, {
                scale: 2.5,
                opacity: 0,
                duration: 0.6,
                ease: 'power2.out',
                onComplete: () => ripple.remove()
            });
        });
    });
});
