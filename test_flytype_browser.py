"""Local browser fixtures for screenshot geometry; no network or brain required."""
import unittest
from playwright.async_api import async_playwright
from flytype import FlyType


class BrowserGeometryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=True)
        self.page = await self.browser.new_page(viewport={"width": 600, "height": 400})
        self.fly = FlyType.__new__(FlyType)
        await self.page.set_content('''<style>
            #clip {position:relative; width:220px; height:100px; overflow:hidden}
            #words {width:220px; height:200px}
            .word {position:absolute; display:flex; height:28px}
            letter {display:block; width:14px; height:28px; color:#666}
            </style><div id="clip"><div id="words"></div></div><div id="result" style="display:none">Done</div>''')
        await self.page.evaluate('''() => {
            const root = document.querySelector('#words');
            for(let i=0;i<8;i++) {
                const w = document.createElement('div'); w.className='word'+(i===0?' active':'');
                w.style.left=(i%2)*90+'px'; w.style.top=Math.floor(i/2)*32+'px';
                w.innerHTML='<letter>a</letter><letter>b</letter>'; root.append(w);
            }
        }''')

    async def asyncTearDown(self):
        await self.browser.close()
        await self.playwright.stop()

    async def test_three_rows_excludes_partly_clipped_fourth_row(self):
        geometry = await self.fly.get_geometry(self.page)
        self.assertEqual([w['word_index'] for w in geometry], list(range(6)))
        self.assertEqual(len({w['row_y'] for w in geometry}), 3)
        self.assertNotIn('char', geometry[0]['boxes'][0])

    async def test_occluded_active_word_cannot_be_skipped(self):
        await self.page.evaluate('''() => {
            const overlay = document.createElement('div');
            overlay.style.cssText='position:absolute;left:8px;top:8px;width:30px;height:28px;z-index:10';
            document.body.append(overlay);
        }''')
        self.assertEqual(await self.fly.get_geometry(self.page), [])

    async def test_stale_word_and_end_of_test_invalidate_cache(self):
        word = (await self.fly.get_geometry(self.page))[0]
        self.assertTrue(await self.fly.cached_word_is_current(self.page, word))
        await self.page.evaluate('''() => {
            const words = document.querySelectorAll('.word');
            words[0].classList.remove('active'); words[1].classList.add('active');
        }''')
        self.assertFalse(await self.fly.cached_word_is_current(self.page, word))
        word = (await self.fly.get_geometry(self.page))[0]
        await self.page.locator('#result').evaluate("el => el.style.display='block'")
        self.assertFalse(await self.fly.cached_word_is_current(self.page, word))

    async def test_blur_cannot_pass_as_stable_rendering(self):
        await self.page.locator('#words').evaluate("el => el.style.filter='blur(3px)'")
        self.assertIsNone(await self.fly.get_geometry(self.page))

    async def test_typed_active_word_keeps_its_space_boundary(self):
        await self.page.locator('.word.active letter').evaluate_all("letters => letters.forEach(el => el.className='correct')")
        geometry = await self.fly.get_geometry(self.page)
        self.assertEqual(geometry[0]['word_index'], 0)
        self.assertEqual(geometry[0]['boxes'], [])


if __name__ == '__main__':
    unittest.main()
